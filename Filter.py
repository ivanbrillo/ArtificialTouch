def alpha_trimmed_mean_filter(data: list, win_size: int, alpha: int) -> list:
    n = len(data)
    result = [0] * (n - win_size)

    assert win_size > alpha > 0 and alpha % 4 == 0 and win_size % 2 == 0

    for i in range(win_size // 2, n - win_size // 2):
        window = data[i - win_size // 2:i + win_size // 2]
        window.sort()
        result[i - win_size // 2] = sum(window[alpha // 4:win_size - (alpha // 4) * 3]) / (win_size - alpha)

    return [result[0]] * (win_size // 2) + result + [result[-1]] * (win_size // 2)


def filter_dataframes(list_df: list, win_size: int, alpha: int):
    for df in list_df:
        df["posz"] = alpha_trimmed_mean_filter(list(df["posz"]), win_size=win_size, alpha=alpha)
        df["Fz"] = alpha_trimmed_mean_filter(list(df["Fz"]), win_size=win_size, alpha=alpha)
