import numpy as np
from stream.node_tensor import NodeTensor


def main() -> None:
    data = np.zeros((1, 32, 32))
    node_tensor = NodeTensor(data, pre_allocation_size=32)
    sliced_tensor = node_tensor.slice(starts=0, ends=16, axis=1)

    print(f"Original shape: {data.shape}")
    print(f"NodeTensor shape: {node_tensor.full_shape}")
    print(f"Sliced shape: {sliced_tensor.full_shape}")


if __name__ == "__main__":
    main()
