using DoubleDouble;
using DoubleDoubleGeometry.Geometry3D;

namespace SphereTriangleDivisionHP;

internal readonly record struct SummaryRow(string File, int N, ddouble MinArea, ddouble MaxArea, ddouble StdArea);
internal readonly record struct HistoryRow(int Iteration, ddouble MeanArea, ddouble StdArea, ddouble MaxRel);
internal readonly record struct DivisionResult(int N, Vector3D[,] Positions);
internal readonly record struct OptimizationResult(Vector3D[,] Positions, TriangleKey[] TriangleKeys, HistoryRow[] History);
internal readonly record struct TriangleKey(LatticeKey A, LatticeKey B, LatticeKey C);
internal readonly record struct LatticeKey(int I, int J);

internal enum ConstraintMode {
    Corner,
    Edge,
    Interior,
}
