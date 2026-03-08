using DoubleDouble;
using System.Globalization;

namespace SphereTriangleDivisionHP;

internal sealed class Config {
    public required string InputDir { get; init; }
    public required string OutputDir { get; init; }
    public required int Iterations { get; init; }
    public required ddouble LearningRate { get; init; }
    public required bool LrDecay { get; init; }
    public required int VerboseEvery { get; init; }
    public required HashSet<int> TargetNs { get; init; }

    public static Config Parse(string[] args, string baseDir) {
        string inputDir = Path.GetFullPath(Path.Combine(baseDir, "../../../../", "results"));
        string outputDir = Path.GetFullPath(Path.Combine(baseDir, "../../../../", "results_high_precision"));

        int iterations = 400000;
        ddouble lr = 0.2d;
        bool lrDecay = true;
        int verboseEvery = 1000;
        HashSet<int> targetNs = [];

        for (int i = 0; i < args.Length; i++) {
            string arg = args[i];
            string next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"missing value for {arg}");

            switch (arg) {
                case "--input-dir":
                    inputDir = Path.GetFullPath(next());
                    break;
                case "--output-dir":
                    outputDir = Path.GetFullPath(next());
                    break;
                case "--iterations":
                    iterations = int.Parse(next(), CultureInfo.InvariantCulture);
                    break;
                case "--lr":
                    lr = ddouble.Parse(next(), CultureInfo.InvariantCulture);
                    break;
                case "--verbose-every":
                    verboseEvery = int.Parse(next(), CultureInfo.InvariantCulture);
                    break;
                case "--no-lr-decay":
                    lrDecay = false;
                    break;
                case "--n":
                    foreach (string token in next().Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)) {
                        targetNs.Add(int.Parse(token, CultureInfo.InvariantCulture));
                    }
                    break;
                default:
                    throw new ArgumentException($"unknown argument: {arg}");
            }
        }

        return new Config {
            InputDir = inputDir,
            OutputDir = outputDir,
            Iterations = iterations,
            LearningRate = lr,
            LrDecay = lrDecay,
            VerboseEvery = verboseEvery,
            TargetNs = targetNs,
        };
    }
}
