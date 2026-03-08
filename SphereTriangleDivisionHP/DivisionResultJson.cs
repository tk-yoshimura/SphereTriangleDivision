using DoubleDouble;
using DoubleDoubleGeometry.Geometry3D;
using System.Globalization;
using System.Text.Json;

namespace SphereTriangleDivisionHP;

internal static class DivisionResultJson {
    public static DivisionResult Load(string path) {
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(path));

        int n = doc.RootElement.GetProperty("N").GetInt32();
        Dictionary<(int, int, int), Vector3D> canonical = [];

        foreach (JsonElement point in doc.RootElement.GetProperty("points").EnumerateArray()) {
            int i = point.GetProperty("i").GetInt32();
            int j = point.GetProperty("j").GetInt32();
            int k = n - i - j;

            JsonElement xyzElem = point.GetProperty("xyz");
            Vector3D xyz = new(
                ReadDDouble(xyzElem[0]),
                ReadDDouble(xyzElem[1]),
                ReadDDouble(xyzElem[2])
            );

            xyz = ApplyIndexAveraging(xyz, i, j, k);

            ((int ci, int cj, int ck) key, Vector3D canonicalVec) = LatticeSymmetry.CanonicalizeTriplet(i, j, k, xyz);
            canonical[key] = canonicalVec;
        }

        Vector3D[,] positions = GridArrayUtil.CreateZeroFilled(n + 1, n + 1);

        foreach (((int i, int j, int k) key, Vector3D xyz) in canonical) {
            int[] idx = [key.i, key.j, key.k];

            foreach (int[] p in LatticeSymmetry.Permutations) {
                int pi = idx[p[0]];
                int pj = idx[p[1]];
                int pk = idx[p[2]];

                if (pi < 0 || pj < 0 || pk < 0 || pi > n || pj > n || pk > n || pi + pj + pk != n) {
                    continue;
                }

                if (pi + pj <= n) {
                    positions[pi, pj] = LatticeSymmetry.Permute(xyz, p);
                }
            }
        }

        return new DivisionResult(n, positions);
    }

    public static void Save(string path, int n, Vector3D[,] positions, bool indexAveraging) {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);

        Dictionary<(int, int, int), Vector3D> canonical = [];

        foreach (LatticeKey key in LatticeGrid.BuildPointKeys(n)) {
            Vector3D xyz = positions[key.I, key.J];
            int k = n - key.I - key.J;
            ((int ci, int cj, int ck) canonicalKey, Vector3D canonicalVec) =
                LatticeSymmetry.CanonicalizeTriplet(key.I, key.J, k, xyz);

            if (!canonical.ContainsKey(canonicalKey)) {
                canonical[canonicalKey] = canonicalVec;
            }
        }

        var pointLines = canonical
            .OrderBy(p => p.Key.Item1)
            .ThenBy(p => p.Key.Item2)
            .ThenBy(p => p.Key.Item3)
            .Select(pair => {
            int i = pair.Key.Item1;
            int j = pair.Key.Item2;
            int k = pair.Key.Item3;

            if (i > j) {
                return null;
            }
            if (j == k && i != j) {
                return null;
            }

            Vector3D v = indexAveraging ? ApplyIndexAveraging(pair.Value, i, j, k) : pair.Value;
            return $$"""
                {
                  "i": {{i}},
                  "j": {{j}},
                  "xyz": [
                    {{FormatDDouble(v.X)}},
                    {{FormatDDouble(v.Y)}},
                    {{FormatDDouble(v.Z)}}
                  ]
                }
                """;
            })
            .Where(text => text is not null)
            .Cast<string>()
            .ToList();

        string json = $$"""
            {
              "N": {{n}},
              "points": [
            {{string.Join(",\n", pointLines.Select(IndentPointJson))}}
              ]
            }
            """;

        File.WriteAllText(path, json + Environment.NewLine);
    }

    private static Vector3D ApplyIndexAveraging(Vector3D v, int i, int j, int k) {
        ddouble x = v.X, y = v.Y, z = v.Z;

        if (i == j && j == k) {
            x = y = z = ddouble.Sqrt(3d) / 3d;
        }
        else if (i == j) {
            x = y = k > 0 ? (x + y) / 2d : ddouble.Sqrt(2d) / 2d;
        }
        else if (j == k) {
            y = z = i > 0 ? (y + z) / 2d : ddouble.Sqrt(2d) / 2d;
        }
        else if (i == k) {
            x = z = j > 0 ? (x + z) / 2d : ddouble.Sqrt(2d) / 2d;
        }

        return new Vector3D(x, y, z);
    }

    private static ddouble ReadDDouble(JsonElement elem) =>
        elem.ValueKind switch {
            JsonValueKind.Number => ddouble.Parse(elem.GetRawText(), CultureInfo.InvariantCulture),
            JsonValueKind.String => ddouble.Parse(elem.GetString()!, CultureInfo.InvariantCulture),
            _ => throw new JsonException($"unsupported number token: {elem.ValueKind}")
        };

    private static string FormatDDouble(ddouble value) {
        if (value == 0) {
            return "0.0";
        }
        if (value == 1) {
            return "1.0";
        }

        return value.ToString();
    }

    private static string IndentPointJson(string text) =>
        string.Join(Environment.NewLine, text.Split('\n').Select(line => $"    {line}"));
}
