#include <ipc/potentials/barrier_potential.hpp>
#include <ipc/smooth_contact/smooth_contact_potential.hpp>
#include <ipc/distance/line_line.hpp>

#include <CLI/CLI.hpp>

#include <igl/read_triangle_mesh.h>
#include <igl/edges.h>
#include <igl/readCSV.h>
#include <ipc/ipc.hpp>

#include <paraviewo/ParaviewWriter.hpp>
#include <paraviewo/VTUWriter.hpp>
#include <paraviewo/HDF5VTUWriter.hpp>

using namespace ipc;

bool has_arg(const CLI::App &command_line, const std::string &value)
{
	const auto *opt = command_line.get_option_no_throw(value.size() == 1 ? ("-" + value) : ("--" + value));
	if (!opt)
		return false;

	return opt->count() > 0;
}

bool load_mesh(
    const std::string& mesh_name,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& E,
    Eigen::MatrixXi& F)
{
    const bool success =
        igl::read_triangle_mesh(mesh_name, V, F);
    if (F.size()) {
        igl::edges(F, E);
    }
    return success && V.size() && F.size() && E.size();
}

int main(int argc, char **argv) {
	CLI::App command_line{"ipc"};

	command_line.ignore_case();
	command_line.ignore_underscore();

	// Eigen::setNbThreads(1);
	size_t max_threads = std::numeric_limits<size_t>::max();
	command_line.add_option("--max_threads", max_threads, "Maximum number of threads");

	std::string input_path = "";
	command_line.add_option("-i,--input", input_path, "Path to the input mesh")->check(CLI::ExistingDirectory | CLI::NonexistentPath);

	double dhat = 1e-2;
	command_line.add_option("--dhat", dhat, "dhat");

	double alpha_n = 1e-2;
	command_line.add_option("--alpha_n", alpha_n, "alpha_n");

	double alpha_t = 5e-1;
	command_line.add_option("--alpha_t", alpha_t, "alpha_t");

	std::string output = "";
	command_line.add_option("-o,--output", output, "VTU file for output gradient");

	const std::vector<std::pair<std::string, spdlog::level::level_enum>>
		SPDLOG_LEVEL_NAMES_TO_LEVELS = {
			{"trace", spdlog::level::trace},
			{"debug", spdlog::level::debug},
			{"info", spdlog::level::info},
			{"warning", spdlog::level::warn},
			{"error", spdlog::level::err},
			{"critical", spdlog::level::critical},
			{"off", spdlog::level::off}};
	spdlog::level::level_enum log_level = spdlog::level::debug;
	command_line.add_option("--log_level", log_level, "Log level")
		->transform(CLI::CheckedTransformer(SPDLOG_LEVEL_NAMES_TO_LEVELS, CLI::ignore_case));

	CLI11_PARSE(command_line, argc, argv);

    const BroadPhaseMethod method{3};

    Eigen::MatrixXd vertices;
    Eigen::MatrixXi edges, faces;
    bool success = load_mesh(input_path, vertices, edges, faces);
    if (!success)
        throw std::runtime_error("Failed to load the mesh!");

    CollisionMesh mesh;
    mesh = CollisionMesh(vertices, edges, faces);

    SmoothCollisions<3> collisions;
    ParameterType param(dhat, alpha_t, 0, alpha_n, 0, 2);
    collisions.build(mesh, vertices, param, false, method);

    SmoothContactPotential<SmoothCollisions<3>> potential(param);

	std::cout << "energy: " << potential(collisions, mesh, vertices) << "\n";

	if (output != "") {
		const Eigen::VectorXd grad_b =
			potential.gradient(collisions, mesh, vertices);
		
		std::cout << "grad norm: " << grad_b.norm() << std::endl;
		
		std::unique_ptr<paraviewo::ParaviewWriter> writer = std::make_unique<paraviewo::VTUWriter>();
		writer->add_field("contact_forces", grad_b);

		writer->write_mesh(
			output,
			mesh.rest_positions(),
			mesh.faces());
	}
}
