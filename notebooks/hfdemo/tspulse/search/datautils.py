import itertools

import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class TransformedSyntheticTestDataset(Dataset):
    """
    Generates augmented query samples by applying temporal shifts, amplitude scaling, and random noise to the original data.
    """

    def __init__(self, train_dataset, max_shift=0.2, max_scale=0.2, noise_ratio=0.1):
        self.data = []
        self.family_match_label = []
        self.finegrained_match_label = []
        self.sequence_length = train_dataset.sequence_length

        assert 0 <= max_shift and max_shift < 1
        assert 0 <= max_scale and max_scale < 1
        assert 0 <= noise_ratio and noise_ratio < 1
        self.max_shift = max_shift  # 0.1 means 10% of 512 -> 50
        self.max_scale = max_scale
        self.noise_ratio = noise_ratio

        max_shift_val = int(round(self.sequence_length * max_shift))

        for idx, sample in enumerate(train_dataset):
            signal = sample["past_values"]
            transformed = []
            for c in range(signal.shape[1]):
                base_pattern = signal[:, c]

                if max_shift > 0:
                    shift_amt = np.random.randint(-max_shift_val, max_shift_val)
                    shifted = np.roll(base_pattern, shift_amt)
                else:
                    shifted = base_pattern

                if max_scale > 0:
                    scale = np.random.uniform(1.0 - max_scale, 1.0 + max_scale)  # -30% or + 30% when max_scale=0.3
                    scaled = shifted * scale
                else:
                    scaled = shifted

                if noise_ratio > 0:
                    noise_level = np.mean(np.absolute(scaled)) * noise_ratio
                    noisy = scaled + noise_level * np.random.randn(len(base_pattern))
                else:
                    noisy = scaled
                transformed.append(noisy)

            transformed_tensor = np.stack(transformed, axis=1).astype(np.float32)
            self.data.append(transformed_tensor)
            self.family_match_label.append(sample["family_match"])
            self.finegrained_match_label.append(sample["finegrained_match"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        past_values = self.data[idx]
        return {
            "past_values": past_values,
            "family_match": self.family_match_label[idx],
            "finegrained_match": self.finegrained_match_label[idx],
        }


class SyntheticCombDataset(Dataset):
    """
    Generates a synthetic dataset. For more details, please refer to https://arxiv.org/pdf/2505.13033
    """

    def __init__(self, sequence_length=512, num_channels=1):
        self.data = []
        self.family_match_label = []
        self.finegrained_match_label = []
        self.sequence_length = sequence_length

        self.max_patterns = 8
        self.ops = ["+", "*"]
        aug_num_per_comb_pattern = 3
        num_freq = 10

        combinations = self._get_comb()
        for comb_idx in range(len(combinations)):  # 56
            for f in range(1, num_freq + 1):
                for c in range(num_channels):
                    a, b, op = combinations[comb_idx]
                    s1, l1 = self._gen_data(a, f, c)
                    s2, l2 = self._gen_data(b, f, c)

                    if op == "+":
                        s = s1 + s2
                        disp_op = "Add"
                    elif op == "*":
                        s = s1 * s2
                        disp_op = "Mul"

                    for j in range(aug_num_per_comb_pattern):
                        aug_s = self._augmentation(s)
                        family_match = f"{l1}-{l2}-{disp_op}"
                        finegrained_match = f"{l1}-{l2}-{disp_op}-{f}"
                        self.data.append(aug_s[:, None].astype(np.float32))
                        self.family_match_label.append(family_match)
                        self.finegrained_match_label.append(finegrained_match)

    def _gen_data(self, pattern_type, f=1, c=0):
        base = np.linspace(0, 2 * np.pi * f, self.sequence_length)

        if pattern_type == 0:
            s = np.sin(base + c)
            l = "sin"
        elif pattern_type == 1:
            s = np.cos(base + c) * (np.sin(base * 0.5))
            l = "modcos"
        elif pattern_type == 2:
            s = np.sign(np.sin(base)) * np.abs(np.cos(base * 2))
            l = "squre_modcos"
        elif pattern_type == 3:
            s = np.exp(-((np.linspace(0, 1, self.sequence_length) - f / 2) ** 2) * 40)
            l = "gaussian_spike"
        elif pattern_type == 4:
            s = np.zeros(self.sequence_length)
            s[:: 10 * f] = 1
            l = "impulse"
        elif pattern_type == 5:
            s = np.cumsum(np.random.randn(self.sequence_length)) + f
            l = "randwalk"
        elif pattern_type == 6:
            s = np.sin(base * (c + 1)) + np.cos(base * 2)
            l = "sincos"
        elif pattern_type == 7:
            s = np.tanh(np.sin(base * 3)) + 0.2 * np.random.randn(self.sequence_length)
            l = "tanhmix"
        else:
            raise RuntimeError(f"invalid pattern_type: {pattern_type}")
        return s, l

    def _augmentation(self, base_pattern, max_scale=0.01, noise_ratio=0.01):
        if max_scale > 0:
            scale = np.random.uniform(1.0 - max_scale, 1.0 + max_scale)  # -30% or + 30% when max_scale=0.3
            scaled = base_pattern * scale
        else:
            scaled = base_pattern

        if noise_ratio > 0:
            noise_level = np.mean(np.absolute(scaled)) * noise_ratio
            noisy = scaled + noise_level * np.random.randn(len(base_pattern))
        else:
            noisy = scaled

        return noisy

    def _get_comb(self):
        numbers = list(range(self.max_patterns))
        products = list(itertools.product(numbers, numbers, self.ops))
        combinations = set()
        for prod in products:
            a, b, op = prod
            # Remove duplicates where (a, b, op) and (b, a, op) are considered the same
            if a == b:
                continue
            if (b, a, op) not in combinations:
                combinations.add((a, b, op))
        return sorted(combinations)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        past_values = self.data[idx]
        return {
            "past_values": past_values,
            "family_match": self.family_match_label[idx],
            "finegrained_match": self.finegrained_match_label[idx],
        }


class UCRDataset(Dataset):
    """
    Creates a real dataset from preprocessed UCR data.
    """

    def __init__(self):
        data = np.load("ucr_for_search.npz")
        self.ts = data["ts"]
        self.names = data["names"]
        self.classes = data["classes"]
        self.sequence_length = self.ts.shape[1]

    def __len__(self):
        return self.ts.shape[0]

    def __getitem__(self, idx):
        n, cl = self.names[idx], self.classes[idx]
        return {
            "past_values": self.ts[idx].astype(np.float32),
            "family_match": n,
            "finegrained_match": f"{n}_{cl}",
        }


class RetrievedData(Dataset):
    """
    A utility class for evaluation process.
    """

    def __init__(self, train_dataset, test_dataset, I_all, level):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.I_all = I_all
        self.level_label = level
        self.label_encoder = LabelEncoder()
        self.encoded_train = self.label_encoder.fit_transform([sample[self.level_label] for sample in train_dataset])
        self.encoded_test = self.label_encoder.transform([sample[self.level_label] for sample in test_dataset])

    def __len__(self):
        return len(self.test_dataset)

    def __getitem__(self, index):
        _I = self.I_all[index]
        label_test = self.encoded_test[index]
        labels_train = np.array([self.encoded_train[i] for i in _I])

        return {
            "label_test": label_test,
            "labels_train": labels_train,
            "n_rel": (label_test == self.encoded_train).sum(),
        }
