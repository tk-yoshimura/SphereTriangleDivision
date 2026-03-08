using DoubleDouble;
using DoubleDoubleGeometry.Geometry3D;

namespace SphereTriangleDivisionHP;

internal static class AreaOptimizer {
    private static readonly ddouble lr_eps = ddouble.Parse("1e-30");
    private static readonly ddouble stagnation_eps = ddouble.Parse("1e-30");

    public static OptimizationResult RunTensionEqualizer(
        int n,
        Vector3D[,] initialPositions,
        int iterations,
        ddouble learningRate,
        bool lrDecay,
        int verboseEvery
    ) {
        LatticeKey[] pointKeys = LatticeGrid.BuildPointKeys(n);
        TriangleKey[] triangleKeys = LatticeGrid.BuildTriangleKeys(n);

        Vector3D[,] positions = GridArrayUtil.ClonePositions(initialPositions);
        List<HistoryRow> history = [];

        ddouble lr = learningRate;
        ddouble stdAreaPrev = ddouble.PositiveInfinity;
        ddouble maxRelPrev = ddouble.PositiveInfinity;

        for (int it = 1; it <= iterations; it++) {
            ddouble[] triAreas = new ddouble[triangleKeys.Length];
            Vector3D[] triCenters = new Vector3D[triangleKeys.Length];

            for (int t = 0; t < triangleKeys.Length; t++) {
                TriangleKey tri = triangleKeys[t];
                Vector3D va = GridArrayUtil.RequirePoint(positions, tri.A);
                Vector3D vb = GridArrayUtil.RequirePoint(positions, tri.B);
                Vector3D vc = GridArrayUtil.RequirePoint(positions, tri.C);

                triAreas[t] = SphericalGeometry.SphericalTriangleArea(va, vb, vc);
                triCenters[t] = SphericalGeometry.Normalize(va + vb + vc);
            }

            ddouble meanArea = triAreas.Average();
            ddouble stdArea = StatisticsUtil.StdDevPopulation(triAreas);
            ddouble maxRel = triAreas.Select(v => ddouble.Abs(v - meanArea) / ddouble.Max(meanArea, SphericalGeometry.AreaEps)).Max();
            history.Add(new HistoryRow(it, meanArea, stdArea, maxRel));

            if (lrDecay && stdArea > 0d && maxRel > 0d &&
                stdAreaPrev / stdArea - 1d <= stagnation_eps &&
                maxRelPrev / maxRel - 1d <= stagnation_eps) {
                lr *= ddouble.Parse("0.99");
            }

            stdAreaPrev = stdArea;
            maxRelPrev = maxRel;

            if (verboseEvery > 0 && (it % verboseEvery == 0 || it == iterations)) {
                Console.WriteLine($"iter={it,5} std={stdArea} max_rel={maxRel} lr={lr:e4}");
            }

            if (it == iterations || lr < lr_eps) {
                break;
            }

            Vector3D[,] moveSum = GridArrayUtil.CreateZeroFilled(n + 1, n + 1);
            int[,] moveCount = new int[n + 1, n + 1];

            for (int t = 0; t < triangleKeys.Length; t++) {
                TriangleKey tri = triangleKeys[t];
                ddouble rel = (triAreas[t] - meanArea) / ddouble.Max(meanArea, SphericalGeometry.AreaEps);
                Vector3D center = triCenters[t];

                AccumulateMove(positions, moveSum, moveCount, tri.A, center, rel, n);
                AccumulateMove(positions, moveSum, moveCount, tri.B, center, rel, n);
                AccumulateMove(positions, moveSum, moveCount, tri.C, center, rel, n);
            }

            Vector3D[,] nextPositions = GridArrayUtil.CreateZeroFilled(n + 1, n + 1);
            foreach (LatticeKey key in pointKeys) {
                Vector3D current = GridArrayUtil.RequirePoint(positions, key);
                if (moveCount[key.I, key.J] > 0) {
                    Vector3D avgMove = moveSum[key.I, key.J] / moveCount[key.I, key.J];
                    nextPositions[key.I, key.J] = ProjectVertex(current + lr * avgMove, key, n);
                }
                else {
                    nextPositions[key.I, key.J] = ProjectVertex(current, key, n);
                }
            }

            positions = nextPositions;
        }

        return new OptimizationResult(positions, triangleKeys, history.ToArray());
    }

    private static void AccumulateMove(Vector3D[,] positions, Vector3D[,] moveSum,
        int[,] moveCount, LatticeKey key, Vector3D center, ddouble rel, int n
    ) {
        Vector3D vertex = GridArrayUtil.RequirePoint(positions, key);
        Vector3D projectedCenter = ProjectCenterForVertex(center, key, n);
        Vector3D delta = rel * (projectedCenter - vertex);

        moveSum[key.I, key.J] = moveSum[key.I, key.J] + delta;
        moveCount[key.I, key.J]++;
    }

    private static Vector3D ProjectVertex(Vector3D v, LatticeKey key, int n) {
        (ConstraintMode mode, int axis) = LatticeGrid.ClassifyVertexConstraint(key, n);

        if (mode == ConstraintMode.Corner) {
            if (key.I == n) {
                return new Vector3D(1d, 0d, 0d);
            }
            if (key.J == n) {
                return new Vector3D(0d, 1d, 0d);
            }
            return new Vector3D(0d, 0d, 1d);
        }

        Vector3D projected = SphericalGeometry.Max(v, 0d);

        if (mode == ConstraintMode.Edge) {
            projected = axis switch {
                0 => new Vector3D(0d, projected.Y, projected.Z),
                1 => new Vector3D(projected.X, 0d, projected.Z),
                _ => new Vector3D(projected.X, projected.Y, 0d),
            };

            if (axis == 0 && projected.Y == 0d && projected.Z == 0d) {
                projected = new Vector3D(0d, ddouble.Sqrt(2d) / 2d, ddouble.Sqrt(2d) / 2d);
            }
            else if (axis == 1 && projected.X == 0d && projected.Z == 0d) {
                projected = new Vector3D(ddouble.Sqrt(2d) / 2d, 0d, ddouble.Sqrt(2d) / 2d);
            }
            else if (axis == 2 && projected.X == 0d && projected.Y == 0d) {
                projected = new Vector3D(ddouble.Sqrt(2d) / 2d, ddouble.Sqrt(2d) / 2d, 0d);
            }
        }

        ddouble norm = projected.Norm;
        if (norm <= 0d) {
            return mode switch {
                ConstraintMode.Edge when axis == 0 => new Vector3D(0d, ddouble.Sqrt(2d) / 2d, ddouble.Sqrt(2d) / 2d),
                ConstraintMode.Edge when axis == 1 => new Vector3D(ddouble.Sqrt(2d) / 2d, 0d, ddouble.Sqrt(2d) / 2d),
                ConstraintMode.Edge => new Vector3D(ddouble.Sqrt(2d) / 2d, ddouble.Sqrt(2d) / 2d, 0d),
                _ => SphericalGeometry.Normalize(new Vector3D(1d, 1d, 1d)),
            };
        }

        return projected / norm;
    }

    private static Vector3D ProjectCenterForVertex(Vector3D center, LatticeKey key, int n) {
        (ConstraintMode mode, int axis) = LatticeGrid.ClassifyVertexConstraint(key, n);
        Vector3D c = SphericalGeometry.Normalize(center);

        if (mode == ConstraintMode.Corner) {
            return ProjectVertex(c, key, n);
        }

        if (mode == ConstraintMode.Edge) {
            c = SphericalGeometry.Max(c, 0d);
            c = axis switch {
                0 => new Vector3D(0d, c.Y, c.Z),
                1 => new Vector3D(c.X, 0d, c.Z),
                _ => new Vector3D(c.X, c.Y, 0d),
            };
            return ProjectVertex(c, key, n);
        }

        return c;
    }
}
