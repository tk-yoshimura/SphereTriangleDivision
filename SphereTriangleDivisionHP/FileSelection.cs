using System.Globalization;

namespace SphereTriangleDivisionHP;

internal static class FileSelection {
    public static IEnumerable<string> EnumerateInputFiles(string inputDir, HashSet<int> targetNs) {
        IEnumerable<string> files = Directory
            .EnumerateFiles(inputDir, "division_result_*.json")
            .OrderBy(path => ExtractSortKey(Path.GetFileNameWithoutExtension(path)));

        if (targetNs.Count > 0) {
            files = files.Where(path => TryExtractN(path, out int n) && targetNs.Contains(n));
        }

        return files;
    }

    private static (int n, string stem) ExtractSortKey(string stem) {
        return TryExtractN(stem, out int n) ? (n, stem) : (int.MaxValue, stem);
    }

    private static bool TryExtractN(string pathOrStem, out int n) {
        string stem = Path.GetFileNameWithoutExtension(pathOrStem);
        string[] parts = stem.Split('_', StringSplitOptions.RemoveEmptyEntries);

        foreach (string part in parts) {
            if (int.TryParse(part, NumberStyles.Integer, CultureInfo.InvariantCulture, out n)) {
                return true;
            }
        }

        n = -1;
        return false;
    }
}
