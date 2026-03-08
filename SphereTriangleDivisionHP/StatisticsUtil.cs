using DoubleDouble;

namespace SphereTriangleDivisionHP;

internal static class StatisticsUtil {
    public static ddouble StdDevPopulation(IReadOnlyList<ddouble> values) {
        ddouble mean = values.Average();
        ddouble squareMean = values.Select(v => ddouble.Square(v - mean)).Average();
        return ddouble.Sqrt(squareMean);
    }
}
