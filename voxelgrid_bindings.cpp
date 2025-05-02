#include "jps_planner/jps_planner.h"
#include "raycast.hpp"
#include "voxel_grid.hpp"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace py = pybind11;

std::shared_ptr<voxel_grid_util::VoxelGrid>
CreateVoxelGridFromConfig(const std::string &config_file_path) {
  YAML::Node config = YAML::LoadFile(config_file_path);
  YAML::Node params = config["env_builder_node"]["ros__parameters"];

  // Read configuration parameters
  std::vector<double> origin_grid_vec =
      params["origin_grid"].as<std::vector<double>>();
  std::vector<double> dimension_grid_vec =
      params["dimension_grid"].as<std::vector<double>>();
  double vox_size = params["vox_size"].as<double>();
  bool free_grid = params["free_grid"].as<bool>();
  bool multi_obst_size = params["multi_obst_size"].as<bool>();
  bool multi_obst_position = params["multi_obst_position"].as<bool>();
  std::vector<double> range_obst_vec =
      params["range_obst"].as<std::vector<double>>();
  std::vector<double> origin_obst_vec =
      params["origin_obst"].as<std::vector<double>>();
  std::vector<double> size_obst_vec =
      params["size_obst"].as<std::vector<double>>();
  int n_obst = params["n_obst"].as<int>();
  int rand_seed = params["rand_seed"].as<int>();
  std::vector<double> position_obst_vec;
  if (params["position_obst_vec"])
    position_obst_vec = params["position_obst_vec"].as<std::vector<double>>();

  // publish_period and other topics are not used in this standalone function

  // Convert vectors to std::array<double,3>
  std::array<double, 3> origin_grid = {origin_grid_vec[0], origin_grid_vec[1],
                                       origin_grid_vec[2]};
  std::array<double, 3> dimension_grid = {
      dimension_grid_vec[0], dimension_grid_vec[1], dimension_grid_vec[2]};
  std::array<double, 3> range_obst = {range_obst_vec[0], range_obst_vec[1],
                                      range_obst_vec[2]};
  std::array<double, 3> origin_obst = {origin_obst_vec[0], origin_obst_vec[1],
                                       origin_obst_vec[2]};
  std::array<double, 3> size_obst = {size_obst_vec[0], size_obst_vec[1],
                                     size_obst_vec[2]};

  // Create empty voxel grid
  Eigen::Vector3d vg_origin(origin_grid[0], origin_grid[1], origin_grid[2]);
  Eigen::Vector3d vg_dimension(dimension_grid[0], dimension_grid[1],
                               dimension_grid[2]);
  auto voxel_grid_ptr = std::make_shared<voxel_grid_util::VoxelGrid>(
      vg_origin, vg_dimension, vox_size, free_grid);

  // Seed the random generator
  std::srand(rand_seed);

  // Determine number of obstacles to add
  int n_obstacles =
      multi_obst_position ? (position_obst_vec.size() / 3) : n_obst;

  // Add obstacles
  for (int i = 0; i < n_obstacles; i++) {
    Eigen::Vector3d center_obst;
    if (multi_obst_position) {
      center_obst = Eigen::Vector3d(position_obst_vec[3 * i],
                                    position_obst_vec[3 * i + 1],
                                    position_obst_vec[3 * i + 2]);
    } else {
      double eps = 0.02;
      center_obst(0) =
          ((std::rand() % static_cast<int>((range_obst[0] + eps) * 100)) /
           100.0) +
          origin_obst[0] - origin_grid[0];
      center_obst(1) =
          ((std::rand() % static_cast<int>((range_obst[1] + eps) * 100)) /
           100.0) +
          origin_obst[1] - origin_grid[1];
      center_obst(2) =
          ((std::rand() % static_cast<int>((range_obst[2] + eps) * 100)) /
           100.0) +
          origin_obst[2] - origin_grid[2];
    }

    Eigen::Vector3d dim_obst;
    if (multi_obst_size) {
      dim_obst = Eigen::Vector3d(size_obst_vec[3 * i], size_obst_vec[3 * i + 1],
                                 size_obst_vec[3 * i + 2]);
    } else {
      dim_obst = Eigen::Vector3d(size_obst[0], size_obst[1], size_obst[2]);
    }
    voxel_grid_util::AddObstacle(voxel_grid_ptr, center_obst, dim_obst);
  }

  return voxel_grid_ptr;
}

std::vector<std::vector<double>>
GetPath(const std::vector<double> &start_arg,
        const std::vector<double> &goal_arg,
        std::shared_ptr<voxel_grid_util::VoxelGrid> vg_util,
        std::vector<std::vector<double>> &path_out, bool verbose) {
  using namespace JPS;

  // Ensure input vectors have correct size
  if (start_arg.size() != 3 || goal_arg.size() != 3) {
    throw std::invalid_argument("Start and goal must have exactly 3 elements.");
  }

  // Mark unknown voxels
  vg_util->SetUnknown(99);

  // Define start and goal
  Eigen::Vector3d start(start_arg[0], start_arg[1], start_arg[2]);
  Eigen::Vector3d goal(goal_arg[0], goal_arg[1], goal_arg[2]);

  // Store map in map_util
  auto map_util = std::make_shared<VoxelMapUtil>();
  map_util->setMap(vg_util->GetOrigin(), vg_util->GetDim(), vg_util->GetData(),
                   vg_util->GetVoxSize());

  auto planner_ptr = std::make_shared<JPSPlanner<3>>(false);
  planner_ptr->setMapUtil(map_util);
  planner_ptr->updateMap();
  bool valid_jps = planner_ptr->plan_occ(start, goal, 1, false);

  // Retrieve path
  auto path_jps = planner_ptr->getRawPath();
  for (const auto &point : path_jps) {
    path_out.push_back({point(0), point(1), point(2)});
  }
  return path_out;
}

bool IsLineClear(const Eigen::Vector3d &pt_start, const Eigen::Vector3d &pt_end,
                 const ::voxel_grid_util::VoxelGrid &vg,
                 const double max_dist_raycast, Eigen::Vector3d &collision_pt,
                 std::vector<Eigen::Vector3d> &visited_points,
                 const bool verbose) {

  // first check if both the start and the end points are inside the grid
  Eigen::Vector3i pt_start_i(pt_start(0), pt_start(1), pt_start(2));
  Eigen::Vector3i pt_end_i(pt_end(0), pt_end(1), pt_end(2));
  /*
  if (!vg.IsInsideGridInt(pt_start_i)) {
    std::cout << "Warning: start of the ray is outside the grid "
              << pt_start_i.transpose() << std::endl;
  }

  if (!vg.IsInsideGridInt(pt_end_i)) {
    std::cout << "Warning: end of the ray is outside the grid "
              << pt_end_i.transpose() << std::endl;
  }
  */

  // Raycast and check if the final position is the same as the collision_pt
  Eigen::Vector3d col_pt(-1, -1, -1);
  visited_points = ::voxel_grid_util::Raycast(pt_start, pt_end, col_pt, vg,
                                              max_dist_raycast, verbose);

  if (col_pt(0) == -1) {
    return true;
  } else {
    collision_pt = col_pt;
    return false;
  }
}

void ClearLine(::voxel_grid_util::VoxelGrid &vg,
               ::voxel_grid_util::VoxelGrid &vg_final,
               const ::Eigen::Vector3d &start, const ::Eigen::Vector3d &end) {
  // only clear the line if is it within the field of view
  bool in_fov = true;

  if (in_fov) {
    ::Eigen::Vector3d collision_pt;
    ::std::vector<::Eigen::Vector3d> visited_points;
    double max_dist_raycast = (start - end).norm();
    bool line_clear = IsLineClear(start, end, vg, max_dist_raycast,
                                  collision_pt, visited_points, false);
    // if line is not clear than the last point is a collision point and we
    // don't need to clear it in the voxel grid
    if (line_clear) {
      visited_points.push_back(end);
    } else {
      ::Eigen::Vector3d last_point = (end - start) * 1e-7 + collision_pt;
      ::Eigen::Vector3i last_point_int(last_point[0], last_point[1],
                                       last_point[2]);
      // check around last_point_int to see the voxels that are occupied;
      vg_final.SetVoxelInt(last_point_int, ENV_BUILDER_OCC);
    }

    int vec_size = visited_points.size();
    for (int i = 0; i < vec_size - 1; i++) {
      vg_final.SetVoxelInt(
          ::Eigen::Vector3i(
              (visited_points[i](0) + visited_points[i + 1](0)) / 2.0,
              (visited_points[i](1) + visited_points[i + 1](1)) / 2.0,
              (visited_points[i](2) + visited_points[i + 1](2)) / 2.0),
          0);
    }
  }
}

void RaycastAndClear(::voxel_grid_util::VoxelGrid &vg,
                     Eigen::Ref<const Eigen::Vector3d> start) {
  /* this implementation is for 360 degrees raycasting i.e. we assume we see 360
  degrees around the drone with cameras; the idea is to raycast to the border
  voxels of the grid; this way we guarantee that we don't miss a voxel while
  maintaining the number of raycasted lines low */

  // get params

  ::Eigen::Vector3d origin = vg.GetOrigin();
  ::Eigen::Vector3i dim = vg.GetDim();

  // create final voxel grid
  ::voxel_grid_util::VoxelGrid vg_final(origin, dim, vg.GetVoxSize(), false);

  // first raycast the ceiling and the floor
  ::std::vector<int> k_vec = {0, dim(2) - 1};
  for (int i = 0; i < dim(0); i++) {
    for (int j = 0; j < dim(1); j++) {
      for (int k : k_vec) {
        ::Eigen::Vector3d end(i + 0.5, j + 0.5, k + 0.5);
        ClearLine(vg, vg_final, start, end);
      }
    }
  }

  // then raycast the wall with fixed y coordinate
  ::std::vector<int> j_vec = {0, dim(1) - 1};
  for (int i = 0; i < dim(0); i++) {
    for (int k = 0; k < dim(2); k++) {
      for (int j : j_vec) {
        ::Eigen::Vector3d end(i + 0.5, j + 0.5, k + 0.5);
        ClearLine(vg, vg_final, start, end);
      }
    }
  }

  // then raycast the wall with fixed x coordinate
  ::std::vector<int> i_vec = {0, dim(0) - 1};
  for (int j = 0; j < dim(1); j++) {
    for (int k = 0; k < dim(2); k++) {
      for (int i : i_vec) {
        ::Eigen::Vector3d end(i + 0.5, j + 0.5, k + 0.5);
        ClearLine(vg, vg_final, start, end);
      }
    }
  }

  // set vg to vg_final
  vg = vg_final;
}

double ComputeMapCompleteness(::voxel_grid_util::VoxelGrid &vg_global) {
  // compute how many unknown cells exist
  ::Eigen::Vector3i dim = vg_global.GetDim();
  double completeness = 0.0;
  double known = 0.0; // unknown cells
  for (int i = 0; i < dim[0]; i++) {
    for (int j = 0; j < dim[1]; j++) {
      for (int k = 0; k < dim[2]; k++) {
        // the voxels of the new voxel grid stay the same unless they are
        // unknown; in that cast we replace them with the values seen in the old
        // voxel grid
        ::Eigen::Vector3i coord(i, j, k);
        if (!vg_global.IsUnknown(coord)) {
          known += 1.0;
        }
      }
    }
  }
  completeness = known / vg_global.GetDataSize();
  return completeness;
}

void MergeVoxelGridsRay(const ::voxel_grid_util::VoxelGrid &vg_old,
                   ::voxel_grid_util::VoxelGrid &vg_new,
                   ::voxel_grid_util::VoxelGrid &count_map) {
  // create final voxel grid

  // update the final voxel grid by going through each voxel and merging the old
  // with the new
  double voxel_size = vg_new.GetVoxSize();
  ::Eigen::Vector3i dim = vg_new.GetDim();
  ::Eigen::Vector3d offset_double = (vg_new.GetOrigin() - vg_old.GetOrigin());
  /* ::std::cout << "vg_new origin:" << vg_new.GetOrigin().transpose() */
  /* << ::std::endl; */
  /* ::std::cout << "vg_old origin:" << vg_old.GetOrigin().transpose() */
  /* << ::std::endl; */
  ::Eigen::Vector3i offset_int;
  offset_int[0] = round(offset_double[0] / voxel_size);
  offset_int[1] = round(offset_double[1] / voxel_size);
  offset_int[2] = round(offset_double[2] / voxel_size);
  /* ::std::cout << "offset_int: " << offset_int.transpose() << ::std::endl; */
  for (int i = 0; i < dim[0]; i++) {
    for (int j = 0; j < dim[1]; j++) {
      for (int k = 0; k < dim[2]; k++) {
        // the voxels of the new voxel grid stay the same unless they are
        // unknown; in that cast we replace them with the values seen in the old
        // voxel grid
        ::Eigen::Vector3i coord(i, j, k);
        ::Eigen::Vector3i coord_final = coord + offset_int;
        if (!vg_old.IsUnknown(coord_final)) {
          count_map.SetVoxelInt(coord, count_map.GetVoxelInt(coord) + 1);
        }
        if (vg_new.IsUnknown(coord)) {
          //::Eigen::Vector3i coord_final = coord + offset_int;
          int8_t vox_value = vg_old.GetVoxelInt(coord_final);
          vg_new.SetVoxelInt(coord, vox_value);
        }
      }
    }
  }
}

int TotalRaycast(const ::voxel_grid_util::VoxelGrid &vg_global,
                 ::voxel_grid_util::VoxelGrid &count_map,
                 Eigen::Ref<const Eigen::Vector3d> pos_curr,
                 ::voxel_grid_util::VoxelGrid &voxel_grid_curr) {

  int newly_discovered = 0;
  double voxel_size = vg_global.GetVoxSize();
  ::Eigen::Vector3d origin_grid = vg_global.GetOrigin();
  //::Eigen::Vector3d voxel_grid_range = voxel_grid_curr.GetRealDim();
  ::Eigen::Vector3d voxel_grid_range;
  voxel_grid_range[0] = 4.0;
  voxel_grid_range[1] = 4.0;
  voxel_grid_range[2] = 4.0;
  ::Eigen::Vector3d origin;

  origin[0] = (pos_curr[0] - voxel_grid_range[0] / 2);
  origin[1] = (pos_curr[1] - voxel_grid_range[1] / 2);
  origin[2] = (pos_curr[2] - voxel_grid_range[2] / 2);
  origin[0] = round((origin[0] - origin_grid[0]) / voxel_size) * voxel_size +
              origin_grid[0];
  origin[1] = round((origin[1] - origin_grid[1]) / voxel_size) * voxel_size +
              origin_grid[1];
  origin[2] = round((origin[2] - origin_grid[2]) / voxel_size) * voxel_size +
              origin_grid[2];
  ::Eigen::Vector3i dim;
  dim[0] = floor(voxel_grid_range[0] / voxel_size);
  dim[1] = floor(voxel_grid_range[1] / voxel_size);
  dim[2] = floor(voxel_grid_range[2] / voxel_size);

  ::std::vector<int> start_idx;
  start_idx.push_back(::std::round((origin[0] - origin_grid[0]) / voxel_size));
  start_idx.push_back(::std::round((origin[1] - origin_grid[1]) / voxel_size));
  start_idx.push_back(::std::round((origin[2] - origin_grid[2]) / voxel_size));
  ::voxel_grid_util::VoxelGrid vg(origin, dim, voxel_size, false);
  ::Eigen::Vector3i dim_env = vg_global.GetDim();

  for (int i = start_idx[0]; i < start_idx[0] + int(dim[0]); i++) {
    for (int j = start_idx[1]; j < start_idx[1] + int(dim[1]); j++) {
      for (int k = start_idx[2]; k < start_idx[2] + int(dim[2]); k++) {
        int i_msg = i - start_idx[0];
        int j_msg = j - start_idx[1];
        int k_msg = k - start_idx[2];
        int idx_env =
            i + int(dim_env[0]) * j + int(dim_env[0]) * int(dim_env[1]) * k;
        int8_t data_val = (i < 0 || j < 0 || k < 0 || i >= dim_env[0] ||
                           j >= dim_env[1] || k >= dim_env[2])
                              ? -1
                              : vg_global.GetData()[idx_env];

        vg.SetVoxelInt(::Eigen::Vector3i(i_msg, j_msg, k_msg), data_val);
        if (data_val == 0) {
          vg.SetVoxelInt(::Eigen::Vector3i(i_msg, j_msg, k_msg), -1);
        }
      }
    }
  }

  if (voxel_grid_curr.GetData().size() == 0) {
    voxel_grid_curr =
        ::voxel_grid_util::VoxelGrid(origin, dim, voxel_size, false);
  }

  ::Eigen::Vector3d pos_curr_local = vg.GetCoordLocal(pos_curr);
  RaycastAndClear(vg, pos_curr_local);
  newly_discovered = int(ComputeMapCompleteness(vg) * vg.GetDataSize());
  // std::cout<<ComputeMapCompleteness(vg)<<"  "<<vg.GetDataSize()<<std::endl;
  MergeVoxelGridsRay(vg, voxel_grid_curr, count_map);
  return newly_discovered;
}

// Fonction qui échantillonne des points le long du chemin avec un espacement

// fixe (sample_distance)
std::vector<Eigen::Vector3d>
SamplePathPoints(const std::vector<std::vector<double>> &raw_path,
                 double sample_distance) {
  std::vector<Eigen::Vector3d> sampled;
  if (raw_path.empty())
    return sampled;

  // Conversion du chemin brut en vecteur d'Eigen::Vector3d
  std::vector<Eigen::Vector3d> points;
  for (const auto &pt : raw_path) {
    if (pt.size() < 3)
      continue;
    points.push_back(Eigen::Vector3d(pt[0], pt[1], pt[2]));
  }
  if (points.empty())
    return sampled;

  // On ajoute le point de départ
  sampled.push_back(points.front());

  // Pour chaque segment entre deux points consécutifs, on interpole des points
  // espacés de sample_distance
  for (size_t i = 0; i < points.size() - 1; i++) {
    Eigen::Vector3d start = points[i];
    Eigen::Vector3d end = points[i + 1];
    Eigen::Vector3d diff = end - start;
    double seg_length = diff.norm();
    if (seg_length == 0)
      continue;
    Eigen::Vector3d dir = diff / seg_length;
    // Commencer à sample_distance, puis ajouter sample_distance jusqu'à ce
    // qu'on atteigne la fin du segment
    double t = sample_distance;
    while (t < seg_length) {
      Eigen::Vector3d sample = start + t * dir;
      sampled.push_back(sample);
      t += sample_distance;
    }
    // Ajout du point de fin du segment
    sampled.push_back(end);
  }
  return sampled;
}


void processVoxel(voxel_grid_util::VoxelGrid &obs,
                  voxel_grid_util::VoxelGrid &count_map,
                  const std::shared_ptr<voxel_grid_util::VoxelGrid> &global_vg,
                  const Eigen::Vector3d &pt_world)
{
  Eigen::Vector3i c;
  c[0] = floor((pt_world[0] - obs.GetOrigin()[0]) / obs.GetVoxSize());
  c[1] = floor((pt_world[1] - obs.GetOrigin()[1]) / obs.GetVoxSize());
  c[2] = floor((pt_world[2] - obs.GetOrigin()[2]) / obs.GetVoxSize());
  int old = count_map.GetVoxelInt(c);
  count_map.SetVoxelInt(c, old+1);
  if (obs.GetVoxelInt(c) == ENV_BUILDER_UNK) {
    // mark occupancy or free
    if (global_vg->GetVoxelInt(c) == ENV_BUILDER_OCC) {
      obs.SetVoxelInt(c, ENV_BUILDER_OCC);
    } else {
      obs.SetVoxelInt(c, ENV_BUILDER_FREE);
      TotalRaycast(*global_vg, count_map, pt_world, obs);
    }
    return;
  }
  return;
}

int countUnknownVoxels(const voxel_grid_util::VoxelGrid &vg) {
  int count = 0;
  const auto &data = vg.GetData();
  for (int k = 0; k < int(data.size()); k++) {
    Eigen::Vector3i coord = vg.IdxToCoord(k);
    if (vg.GetVoxelInt(coord) == ENV_BUILDER_UNK) {
      count++;
    }
  }
  return count;
}


py::tuple step_cpp(const std::vector<std::vector<double>> &drone_positions_,
                   const std::vector<std::vector<double>> &actions_,
                   voxel_grid_util::VoxelGrid &observation,
                   voxel_grid_util::VoxelGrid &count_map,
                   std::shared_ptr<voxel_grid_util::VoxelGrid> global_vg,
                   int voxels_per_step)
{
  int num_drones = int(drone_positions_.size());
  std::vector<Eigen::Vector3d> new_positions;
  int reward_truncated = 0;
  int reward_full_path = 0;

  // For each drone
  for (int d = 0; d < num_drones; d++)
  {
    // 1) Plan path
    std::vector<std::vector<double>> path_raw;
    try {
      path_raw = GetPath(drone_positions_[d], actions_[d], global_vg, path_raw, false);
    } catch (const std::exception &e) {
      new_positions.push_back(Eigen::Vector3d(drone_positions_[d][0],
                                              drone_positions_[d][1],
                                              drone_positions_[d][2]));
      continue;
    }
    if (path_raw.empty()) {
      new_positions.push_back(Eigen::Vector3d(drone_positions_[d][0],
                                              drone_positions_[d][1],
                                              drone_positions_[d][2]));
      continue;
    }

    // 2) Sample points
    std::vector<Eigen::Vector3d> sampled = SamplePathPoints(path_raw, 1.0);
    int steps = int(sampled.size());

    // 3) Copy observation for full-path sim

    // 4) Truncated loop on real observation
    for (int i = 0; i < std::min(steps, voxels_per_step); i++) {
    //for (int i = 0; i < steps; i++) {  
      // count old unknowns
      //int nb_unk_old = countUnknownVoxels(observation);

      // process
      processVoxel(observation, count_map, global_vg, sampled[i]);

      // your preserved block: count new unknowns
      //int nb_unk = countUnknownVoxels(observation);
      //reward_truncated += (nb_unk_old - nb_unk);
    }
    /*
    std::vector<voxel_grid_util::voxel_data_type> data_copy = observation.GetData();
    voxel_grid_util::VoxelGrid obs_copy(
        observation.GetOrigin(),
        observation.GetDim(),
        observation.GetVoxSize(),
        data_copy
    );
    
    // 5) Full-path loop on the copy, with “count unknowns” block
    for (int i = voxels_per_step; i < steps; i++) {
      // count old unknowns
      int nb_unk_old = countUnknownVoxels(obs_copy);

      // process on the copy
      processVoxel(obs_copy, global_vg, sampled[i]);

      // your preserved block: count new unknowns
      int nb_unk = countUnknownVoxels(obs_copy);
      reward_full_path += (nb_unk_old - nb_unk);
      
    }
    */
    

    // 6) Advance new position according to truncated motion
    if (steps == 0) {
      new_positions.push_back(Eigen::Vector3d(drone_positions_[d][0],
                                              drone_positions_[d][1],
                                              drone_positions_[d][2]));
    } else {

      int last_i = std::min(steps, voxels_per_step) - 1;
      new_positions.push_back(sampled[last_i]);
      //new_positions.push_back(sampled[steps - 1]);
    }
  }

  // Update drone_positions for Python
  std::vector<std::vector<double>> drone_positions_vector;
  for (auto &v : new_positions) {
    drone_positions_vector.push_back({v[0], v[1], v[2]});
  }

  // Compute completeness & done
  bool done = (ComputeMapCompleteness(observation) >= 0.95);

  // Build Python return values
  py::dict obs_dict;
  obs_dict["observation"] = observation;
  obs_dict["drone_positions"] = drone_positions_vector;

  py::dict info;
  return py::make_tuple(obs_dict, count_map, done, info);
}


PYBIND11_MODULE(voxelgrid, m) {
  // VoxelGrid bindings
  py::class_<voxel_grid_util::VoxelGrid,
             std::shared_ptr<voxel_grid_util::VoxelGrid>>(m, "VoxelGrid")
      .def(py::init<>()) // Default constructor
      .def(py::init<Eigen::Ref<const Eigen::Vector3d>,
                    Eigen::Ref<const Eigen::Vector3i>, double,
                    bool>()) // Grid constructor
      .def("is_inside_grid_int",
           py::overload_cast<const Eigen::Vector3i &>(
               &voxel_grid_util::VoxelGrid::IsInsideGridInt, py::const_))
      .def("is_inside_grid_local",
           &voxel_grid_util::VoxelGrid::IsInsideGridLocal)
      .def("get_dim", &voxel_grid_util::VoxelGrid::GetDim)
      .def("get_real_dim", &voxel_grid_util::VoxelGrid::GetRealDim)
      .def("get_origin", &voxel_grid_util::VoxelGrid::GetOrigin)
      .def("get_data", &voxel_grid_util::VoxelGrid::GetData)
      .def("is_inside_grid_global",
           &voxel_grid_util::VoxelGrid::IsInsideGridGlobal)
      .def("set_voxel_int", &voxel_grid_util::VoxelGrid::SetVoxelInt)
      .def("set_voxel_double", &voxel_grid_util::VoxelGrid::SetVoxelDouble)
      .def("get_voxel_int",
           py::overload_cast<Eigen::Ref<const Eigen::Vector3i>>(
               &voxel_grid_util::VoxelGrid::GetVoxelInt, py::const_))
      .def("get_voxel_int",
           py::overload_cast<Eigen::Ref<const Eigen::Vector3i>>(
               &voxel_grid_util::VoxelGrid::GetVoxelInt, py::const_),
           py::arg("coord_int"),
           "Get voxel value from integer coordinates (local)")
      .def("get_voxel_int",
           py::overload_cast<Eigen::Ref<const Eigen::Vector3d>>(
               &voxel_grid_util::VoxelGrid::GetVoxelInt, py::const_),
           py::arg("coord_double"),
           "Get voxel value from double coordinates (local)");

  // JPSPlanner3D bindings
  py::class_<JPSPlanner<3>, std::shared_ptr<JPSPlanner<3>>>(m, "JPSPlanner3D")
      .def(py::init<bool>(),
           py::arg("verbose") = false) // Constructor with optional verbose
      .def("set_map_util", &JPSPlanner<3>::setMapUtil)
      .def("status", &JPSPlanner<3>::status)
      .def("get_path", &JPSPlanner<3>::getPath)
      .def("get_raw_path", &JPSPlanner<3>::getRawPath)
      .def("remove_line_pts", &JPSPlanner<3>::removeLinePts)
      .def("remove_corner_pts", &JPSPlanner<3>::removeCornerPts)
      .def("update_map", &JPSPlanner<3>::updateMap)
      .def("plan", &JPSPlanner<3>::plan)
      .def("get_open_set", &JPSPlanner<3>::getOpenSet)
      .def("get_close_set", &JPSPlanner<3>::getCloseSet)
      .def("get_all_set", &JPSPlanner<3>::getAllSet);

  // Expose GetPath function to Python
  m.def("get_path", &GetPath, py::arg("start"), py::arg("goal"),
        py::arg("vg_util"), py::arg("path_out"), py::arg("verbose") = false);

  // Raycasting bindings
  m.def("signum", &voxel_grid_util::signum, "Return the sign of a number");
  m.def("mod", &voxel_grid_util::mod, "Find the modulus of a value");
  m.def("intbound", &voxel_grid_util::intbound,
        "Find the smallest positive t such that s + t*ds is an integer");

  m.def("raycast", &voxel_grid_util::Raycast,
        "Raycast between two points and return the raycasted path in a voxel "
        "grid",
        py::arg("start"), py::arg("end"), py::arg("collision_pt"),
        py::arg("vg"), py::arg("max_dist"), py::arg("verbose") = false);

  // Expose RaycastAndClear function to Python
  m.def("raycast_and_clear", &RaycastAndClear,
        "Raycast and clear voxels in a voxel grid from a start position",
        py::arg("vg"), py::arg("start"));

  // Expose MergeVoxelGridsRay function to Python
  m.def("merge_voxel_grids", &MergeVoxelGridsRay,
        "Merge two voxel grids and return the result", py::arg("vg_old"),
        py::arg("vg_new"), py::arg("count_map"));

  m.def("step_cpp", &step_cpp, py::arg("drone_positions"), py::arg("actions"),
        py::arg("observation"), py::arg("count_map"), py::arg("global_vg"),
        py::arg("voxels_per_step"),
        "Exécute un pas dans l'environnement et renvoie (observation, reward, "
        "done, info)");

  m.def("create_voxelgrid_from_config", &CreateVoxelGridFromConfig,
        py::arg("config_file_path"),
        "Creates a voxel grid with obstacles from a YAML configuration file.");

  m.def("get_data_np", [](const voxel_grid_util::VoxelGrid &vg) {
    // Assume vg.GetData() returns a std::vector<int8_t>
    auto data = vg.GetData();
    std::vector<ssize_t> shape = {static_cast<ssize_t>(vg.GetDim()(0)),
                                  static_cast<ssize_t>(vg.GetDim()(1)),
                                  static_cast<ssize_t>(vg.GetDim()(2))};

    std::vector<ssize_t> strides = {
        static_cast<ssize_t>(sizeof(int8_t)) * shape[1] * shape[2],
        static_cast<ssize_t>(sizeof(int8_t)) * shape[2],
        static_cast<ssize_t>(sizeof(int8_t))};

    return py::array(py::buffer_info(
        data.data(),                             // Pointer to buffer
        sizeof(int8_t),                          // Size of one element
        py::format_descriptor<int8_t>::format(), // Python struct-style format
                                                 // descriptor
        3,                                       // Number of dimensions
        shape,                                   // Buffer dimensions
        strides // Strides (in bytes) for each index
        ));
  });
}
