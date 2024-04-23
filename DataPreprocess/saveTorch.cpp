//
// Created by jimmy on 23/10/2023.
//
#include <iostream>
#include <string>
#include <vector>
#include <torch/torch.h>
#include <filesystem>

struct TrajPoint{
    std::string a;
    std::string b;
    torch::Tensor tensor;
};

void printTensor(torch::Tensor &tensor){
    std::cout << tensor << std::endl;
}

int main(){
    auto tensor = torch::zeros({ 5, 5 });
    for (int r = 0; r < 5; r ++){
        for (int c = 0; c < 5; c ++ ){
            tensor[r][c] = r * 5 + c;
        }
    }

    auto rand = torch::rand({5, 5});

    std::vector<torch::Tensor> tensor_arr{tensor, rand};

    auto cat_tensor = torch::stack(tensor_arr, 0);

    //std::tuple<std::string, std::string, torch::Tensor> p1("abc", "defg", tensor);
    c10::Dict<std::string, std::tuple<std::string, std::vector<torch::Tensor>>> p1;
//    p1.insert_or_assign("AAA", std::tuple<std::string, std::vector<torch::Tensor>>("aaa", ));

    auto pickled = torch::pickle_save(p1);
    std::ofstream fout("input.pt", std::ios::out | std::ios::binary);
    fout.write(pickled.data(), pickled.size());
    fout.write(pickled.data(), pickled.size());
    fout.close();
};
