#include <fstream>
#include <set>

#include "portfolio/cli.hpp"
#include "portfolio/clustering.hpp"
#include "portfolio/registry.hpp"
using namespace portfolio;
int main(int argc, char** argv) {
  return guarded([&] {
    Args args(argc, argv);
    args.only({"data", "clusters", "epsilon", "minimum-points", "artifacts", "output", "help"});
    if (args.has("help")) {
      std::cout << "--data CSV --clusters 3 --epsilon 1.2 --minimum-points 5 --output "
                   "artifacts/clusters.csv\n";
      return;
    }
    auto data = read_dataset(args.get("data", "data/seeds.csv"));
    auto scaler = Standardizer::fit(data.x);
    auto x = scaler.transform(data.x);
    int k = args.integer("clusters", 3);
    auto partitions = kmeans(x, k);
    auto density = dbscan(x, number(args.get("epsilon", "1.2")), args.integer("minimum-points", 5));
    auto mixture = gaussian_mixture(x, k);
    auto projection = pca(x);
    auto json_labels = [](const Eigen::VectorXi& values) {
      return Json(std::vector<int>(values.data(), values.data() + values.size()));
    };
    Json metrics = {{"kmeans_ari", adjusted_rand(data.y, partitions.labels)},
                    {"gmm_diagonal_ari", adjusted_rand(data.y, mixture)},
                    {"dbscan_ari_including_noise", adjusted_rand(data.y, density)},
                    {"dbscan_noise_fraction", (density.array() == -1).cast<double>().mean()},
                    {"kmeans_inertia", partitions.objective}};
    std::filesystem::path output = args.get("output", "artifacts/clusters.csv");
    if (!output.parent_path().empty())
      std::filesystem::create_directories(output.parent_path());
    std::ofstream file(output);
    if (!file)
      throw std::runtime_error("Cannot write cluster projection");
    file << "pc1,pc2,kmeans,dbscan,gmm,label\n";
    for (Eigen::Index i = 0; i < x.rows(); ++i)
      file << projection(i, 0) << ',' << projection(i, 1) << ',' << partitions.labels(i) << ','
           << density(i) << ',' << mixture(i) << ',' << data.y(i) << '\n';
    auto artifact = save_artifact(
        args.get("artifacts", "artifacts"), "clustering",
        {{"kind", "exploratory_cluster_assignments"},
         {"normalizer", scaler.json()},
         {"kmeans_labels", json_labels(partitions.labels)},
         {"dbscan_labels", json_labels(density)},
         {"gmm_labels", json_labels(mixture)}},
        {{"metrics", metrics}, {"source_sha256", sha256(args.get("data", "data/seeds.csv"))}});
    print({{"metrics", metrics},
           {"projection", output.string()},
           {"artifact", artifact.path.string()}});
  });
}
