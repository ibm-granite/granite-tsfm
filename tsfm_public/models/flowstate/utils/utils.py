# Copyright contributors to the TSFM project
#
BASE_SEASON = 24.0


def get_fixed_factor(freq: str, domain=None):
    has_weekly = domain in [
        "Transport",
        "Healthcare",
        "Sales",
    ]  # only human-rythm dependent domains have weekly cylces (not for example nature)
    # make different groups, and normalize relative to group
    if freq == "4S":  ############### sub any reasonable cycle group --> Hourly
        factor = BASE_SEASON / (3600.0 / 4)
    elif freq == "10S":
        factor = BASE_SEASON / 360
    elif freq == "T":  ############ sub day group
        factor = BASE_SEASON / (24.0 * 60)  # [24*60]
    elif freq[-1] == "T":
        n_min = int(freq[:-1])
        factor = BASE_SEASON / (24 * 60 / n_min)
    elif freq in ["H", "h"]:
        factor = BASE_SEASON / 24
    elif freq == "6H":
        factor = BASE_SEASON / 4  # only CMIP6 in pretraining --> 24. / 4 or 24 / 365*4 would be better!
    elif freq == "D":
        if has_weekly:
            factor = BASE_SEASON / 7
        else:
            factor = BASE_SEASON / 365  ############## sub year group
    elif freq[-1] == "D" and "WED" not in freq:
        n = int(freq[:-1])
        if has_weekly:
            factor = BASE_SEASON / 7
        else:
            factor = BASE_SEASON / 365  ############## sub year group
        factor *= n
    elif freq == "W" or "W-" in freq:
        factor = BASE_SEASON / (365.0 / 7)
    elif freq == "M" or "M-" in freq or freq.startswith("M"):
        factor = BASE_SEASON / 12
    elif "Q" in freq:
        factor = BASE_SEASON / 4.0  # 'Q' or 'Q-Month'
    elif "A" in freq:
        factor = BASE_SEASON / 4.0  # leap year ??
    else:
        raise NotImplementedError("{freq} not implemented. Add {freq} option to this method")
    return factor
