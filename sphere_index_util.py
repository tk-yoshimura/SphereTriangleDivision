def validate_n(n):
    if n < 1:
        raise ValueError("N must be >= 1.")


def iter_valid_ij(n):
    for i in range(n + 1):
        for j in range(n + 1 - i):
            yield i, j


def k_from_ij(n, i, j):
    return n - i - j


def full_point_count(n):
    return (n + 1) * (n + 2) // 2


def compact_point_count(n):
    # Number of integer triples (i, j, k) with i+j+k=n and i<=j<=k.
    compact_count = (n * (n + 6) + 12) // 12

    return compact_count
