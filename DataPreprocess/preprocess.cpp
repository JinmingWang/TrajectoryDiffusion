#include <iostream>
#include <string>
#include <vector>
#include <torch/torch.h>
#include <filesystem>

/* torch::pickle_save() supporting data container:
 *
 * C++              Python
 * ---              ------
 * std::vector      list
 * std::tuple       tuple
 * c10::Dict        dict
 * torch::Tensor    torch.Tensor
 */

// A trajectory contains many points, each point is a tuple of (time, lat, lon)
typedef std::vector<std::tuple<long, double, double>> DynamicTrajPoints;
// A trajectory metadata is a tuple of (driver_id, points)
typedef std::pair<std::string, DynamicTrajPoints> DynamicTrajMeta;
// A batch of trajectories is a dict of {order_id: (driver_id, points)}
typedef std::map<std::string, DynamicTrajMeta> DynamicDataBatch;

// A static trajectory metadata is a tuple of (order_id, driver_id, lat&lon_tensor, time_tensor)
typedef std::tuple<std::string, std::string, torch::Tensor, torch::Tensor> StaticTrajMeta;
// A static batch is a list of static trajectory metadata
typedef std::vector<StaticTrajMeta> StaticDataBatch;


struct LineInfo{
    std::string driver_id;
    std::string order_id;
    long time;
    double lon;
    double lat;
};


void printTensor(torch::Tensor &tensor){
    std::cout << tensor << std::endl;
}

void printTensors(std::vector<torch::Tensor> &points){
    for (auto &point: points){
        printTensor(point);
    }
}


LineInfo parseLine(std::string &line){
    LineInfo info;
    std::istringstream ss(line);
    std::string token;

    std::getline(ss, info.driver_id, ',');
    std::getline(ss, info.order_id, ',');
    std::getline(ss, token, ',');
    info.time = std::stol(token);
    std::getline(ss, token, ',');
    info.lon = std::stod(token);
    std::getline(ss, token, ',');
    info.lat = std::stod(token);

    return info;
}


StaticTrajMeta makeStaticTraj(const std::string &order_id, DynamicTrajMeta &dynamic_traj_meta){
    long n_points = static_cast<long>(dynamic_traj_meta.second.size());

    auto f64_options = c10::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor lon_lat_tensor = torch::zeros({n_points, 2}, f64_options);

    auto ul_options = c10::TensorOptions().dtype(torch::kInt64);
    torch::Tensor time_tensor = torch::zeros({n_points}, ul_options);

    int pi = 0;
    for (auto &point : dynamic_traj_meta.second){
        time_tensor[pi] = std::get<0>(point);
        lon_lat_tensor[pi][0] = std::get<1>(point);
        lon_lat_tensor[pi][1] = std::get<2>(point);
        pi ++;
    }

    // Create an index tensor and sort it based on time_tensor values
    torch::Tensor indices = torch::argsort(time_tensor);

    // Sort lon_lat_tensor and time_tensor using the sorted indices
    lon_lat_tensor = lon_lat_tensor.index_select(0, indices);
    time_tensor = time_tensor.index_select(0, indices);

    return StaticTrajMeta(order_id, dynamic_traj_meta.first, lon_lat_tensor, time_tensor);
}


void parseFile(std::string &file_path, std::ifstream &file_stream){
    /* dynamic_dataset = [batch_0, batch_1, batch_2, ...]
     * batch_i = {order_0: (driver_id, points), order_1: (driver_id, points), ...}
     * points = tensor of size (N_points, 3)
     * */
    DynamicDataBatch dynamic_dataset;
    std::vector<std::string> order_ids;

    // Read and parse lines
    long line_count = 0;
    long traj_count = 0;
    std::string line;

    // Iterate over all lines, parse each line, add data to dataset
    while (std::getline(file_stream, line)) {
        // print line number and \r
        if (line_count++ % 10000 == 0) std::cout << "\rLine " << line_count << std::flush;
        // parse line
        LineInfo info;
        try{
            info = parseLine(line);
        } catch (...) {
            std::cout << "Error parsing: " << line << std::endl;
            continue;
        }

        // if order exists then append point, if not then add order & driver & first point
        if (dynamic_dataset.find(info.order_id) != dynamic_dataset.end()){
            std::get<1>(dynamic_dataset[info.order_id]).emplace_back(info.time, info.lon, info.lat);
        } else{
            std::tuple<unsigned long, double, double> new_point(info.time, info.lon, info.lat);
            order_ids.emplace_back(info.order_id);
            dynamic_dataset[info.order_id] = DynamicTrajMeta(info.driver_id, DynamicTrajPoints{new_point});
            traj_count ++;
        }
    }

    file_stream.close();
    std::cout << "\nFile parsing done" << std::endl;
    std::cout << "Trajectory Count: " << dynamic_dataset.size() << std::endl;

    std::cout << "Concatenating Tensors" << std::endl;
    StaticDataBatch static_dataset;
    // Iterate over all order, stack traj points into a tensor

    #pragma omp parallel for default(none) shared(static_dataset, order_ids, dynamic_dataset, traj_count, std::cout) num_threads(8)
    for (long i = 0; i < traj_count; i++){
        StaticTrajMeta static_traj_meta = makeStaticTraj(order_ids[i], dynamic_dataset[order_ids[i]]);

        #pragma omp critical
        {
            static_dataset.emplace_back(static_traj_meta);
        }
    }

    std::cout << "Concatenating done, saving\n" << std::endl;
    auto pickled = torch::pickle_save(static_dataset);
    std::ofstream fout(file_path + ".pt", std::ios::out | std::ios::binary);
    fout.write(pickled.data(), pickled.size());
    fout.close();

//
//    std::cout << "Tensor saved." << std::endl;
}



int main() {
    std::string data_dir = "/media/jimmy/MyData/Data/Didi/xian/nov";

    // Iterate over all files under data_dir
    for (const auto& entry : std::filesystem::directory_iterator(data_dir)) {
        std::string file_name = entry.path().filename().string();

        // if file is gpd_YYYYMMDD file (contains trajectory data), then parse it
        if (file_name[0] == 'g' && file_name[file_name.size()-1] != 't') {
            std::string file_path = entry.path().string();

            // Open file as stream
            std::ifstream file_stream(file_path);
            if (file_stream.is_open()) {
                std::cout << "Parsing file " << file_path << std::endl;
                parseFile(file_path, file_stream);
            } else {
                std::cerr << "Error: Unable to open file " << file_path << std::endl;
            }
        }
    }

    return 0;
}
