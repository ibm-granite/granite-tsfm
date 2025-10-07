#!/bin/env bash
set -euo pipefail
IFS=$'\n\t'

echo "Deployment checklist:"
echo "✅ You pulled upstream changes (git fetch upstream && git merge upstream/destiny)."
echo "✅ You've confirmed that you recursively cloned this repository so that NGP-runbook directory contains product security scripts."
echo "✅ You've done a minimal test of the image via make test_local && test_image."
echo "   (remember to run make stop_service_local and/or stop_service_image if tests fail)."
echo "✅ You've pushed all your merges to the origin (aka the private copy of granite-tsfm) (git push origin destiny) after confirming tests pass."
echo "Press any key to continue"
read

# --- check for uncommitted changes ---
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "❌ You have uncommitted changes."
    echo "Please commit or stash them first."
    exit 1
fi

echo "✅ Working tree clean."

# --- ensure required environment variables are set ---
: "${RIS3_IBM_API_KEY:?Environment variable RIS3_IBM_API_KEY must be set}"

# --- show latest tag and get a new one ---
echo "The last tag created is:"
last_tag=$(git for-each-ref --sort=-creatordate --format '%(refname:short)' refs/tags | head -n1)
echo "  $last_tag"
echo
read -rp "Enter a new tag: " tag

if [[ -z $tag ]]; then
    echo "❌ Tag cannot be empty."
    exit 1
fi

# --- build the image ---
echo "🛠️ Building container image..."
SKIP_GPU_BUILD=1 CONTAINER_BUILDER=podman make image

# --- tag image ---
image_local="localhost/tsfminference-cpu:latest"
image_remote="us.icr.io/fctkstus/tsfminference-cpu:${tag}"

echo "🔖 Tagging image as $image_remote"
podman tag "$image_local" "$image_remote"

# --- authenticate and push ---
echo "🔐 Logging into IBM Cloud..."
ibmcloud config --check-version=false
ibmcloud login --apikey "$RIS3_IBM_API_KEY" -r "us-south"

echo "🔐 Logging into IBM Container Registry..."
ibmcloud cr login --client podman

echo "🚀 Pushing image to $image_remote"
podman push "$image_remote"

# --- create and push git tag ---
echo "🏷️  Creating git tag '$tag'"
git tag -a "$tag" -m "Release $tag"

echo "📤 Pushing git tag to origin..."
git push origin "$tag"

echo "✅ Done!"
