using DoubleDouble;

namespace SphereTriangleDivisionHP;

internal static class Program {
    private static int Main(string[] args) {
        Config config = Config.Parse(args, Directory.GetCurrentDirectory());

        if (!Directory.Exists(config.InputDir)) {
            Console.Error.WriteLine($"input directory not found: {config.InputDir}");
            return 1;
        }

        Directory.CreateDirectory(config.OutputDir);

        List<string> inputFiles = FileSelection.EnumerateInputFiles(config.InputDir, config.TargetNs).ToList();
        if (inputFiles.Count == 0) {
            Console.Error.WriteLine("no input JSON files matched.");
            return 1;
        }

        List<SummaryRow> summary = [];

        foreach (string inputPath in inputFiles) {
            string name = Path.GetFileName(inputPath);
            DivisionResult input = DivisionResultJson.Load(inputPath);

            Console.WriteLine($"\n===== {name} / N={input.N} / iterations={config.Iterations} =====");

            OptimizationResult optimized = AreaOptimizer.RunTensionEqualizer(
                input.N,
                input.Positions,
                config.Iterations,
                config.LearningRate,
                config.LrDecay,
                config.VerboseEvery
            );

            ddouble[] areas = SphericalGeometry.ComputeTriangleAreas(optimized.Positions, optimized.TriangleKeys);
            string outputPath = Path.Combine(config.OutputDir, name);

            DivisionResultJson.Save(outputPath, input.N, optimized.Positions, indexAveraging: true);

            ddouble areaMin = areas.Min();
            ddouble areaMax = areas.Max();
            ddouble areaStd = StatisticsUtil.StdDevPopulation(areas);

            Console.WriteLine($"saved result json : {outputPath}");
            Console.WriteLine($"area stats        : min={areaMin}, max={areaMax}, std={areaStd}");

            summary.Add(new SummaryRow(name, input.N, areaMin, areaMax, areaStd));
        }

        Console.WriteLine("\n=== Summary (after high-precision optimization) ===");
        foreach (SummaryRow row in summary) {
            Console.WriteLine($"{row.File}: N={row.N}, min={row.MinArea}, max={row.MaxArea}, std={row.StdArea}");
        }

        return 0;
    }
}
