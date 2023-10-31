import matplotlib.pyplot as plt
import numpy as np
import torch


def saveModel(model:torch.nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


def loadModel(model:torch.nn.Module, path: str) -> None:
    model.load_state_dict(torch.load(path))


def exportONNX(model: torch.nn.Module, sample_inputs: list[torch.Tensor], path: str) -> None:
    model.eval()
    torch.onnx.export(model, sample_inputs, path, verbose=True, opset_version=13, 
                      input_names=['input_traj', "time", "attr"], output_names=['output_traj'])
    

def visualizeTraj(traj: torch.Tensor) -> None:
    """ draw trajectory

    :param traj: trajectory to draw, shape: (2, N) for lat lon
    :return: None
    """
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.plot(traj[0], traj[1], color='blue', marker='o', linestyle='solid', linewidth=1, markersize=1)
    plt.show()



class MovingAverage:
    def __init__(self, window_size: int) -> None:
        self.window = np.zeros(window_size, dtype=np.float32)
        self.window_size = window_size
        self.cursor = 0

    def __lshift__(self, number: float) -> None:
        self.window[self.cursor] = number
        self.cursor = (self.cursor + 1) % self.window_size

    def __str__(self) -> str:
        return str(self.window.mean())
    
    def __repr__(self) -> str:
        return str(self.window.mean())
    
    def __format__(self, format_spec: str) -> str:
        return self.window.mean().__format__(format_spec)

    def __float__(self) -> float:
        return float(self.window.mean())

if __name__ =="__main__":
    ma = MovingAverage(5)
    for i in range(10):
        ma << i
        print(f"{ma:.4f}")