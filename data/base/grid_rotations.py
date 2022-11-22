import torch
import numpy as np
import copy


def grid_rot90(m, i=None):
    if i is None: i = np.random.randint(24)
    if i == 0 : return m
    if i == 1 : return np.rot90(m,                      1, (0, 2))
    if i == 2 : return np.rot90(m,                      2, (0, 2))
    if i == 3 : return np.rot90(m,                      3, (0, 2))
    if i == 4 : return np.rot90(m,                      1, (1, 2))
    if i == 5 : return np.rot90(m,                      1, (2, 1))
    if i == 6 : return          np.rot90(m, 1, (0, 1))
    if i == 7 : return np.rot90(np.rot90(m, 1, (0, 1)), 1, (0, 2))
    if i == 8 : return np.rot90(np.rot90(m, 1, (0, 1)), 2, (0, 2))
    if i == 9 : return np.rot90(np.rot90(m, 1, (0, 1)), 3, (0, 2))
    if i == 10: return np.rot90(np.rot90(m, 1, (0, 1)), 1, (1, 2))

    if i == 11: return np.rot90(np.rot90(m, 1, (0, 1)), 1, (2, 1))
    if i == 12: return          np.rot90(m, 2, (0, 1))
    if i == 13: return np.rot90(np.rot90(m, 2, (0, 1)), 1, (0, 2))
    if i == 14: return np.rot90(np.rot90(m, 2, (0, 1)), 2, (0, 2))
    if i == 15: return np.rot90(np.rot90(m, 2, (0, 1)), 3, (0, 2))
    if i == 16: return np.rot90(np.rot90(m, 2, (0, 1)), 1, (1, 2))
    if i == 17: return np.rot90(np.rot90(m, 2, (0, 1)), 1, (2, 1))
    if i == 18: return          np.rot90(m, 3, (0, 1))
    if i == 19: return np.rot90(np.rot90(m, 3, (0, 1)), 1, (0, 2))
    if i == 20: return np.rot90(np.rot90(m, 3, (0, 1)), 2, (0, 2))
    if i == 21: return np.rot90(np.rot90(m, 3, (0, 1)), 3, (0, 2))
    if i == 22: return np.rot90(np.rot90(m, 3, (0, 1)), 1, (1, 2))
    if i == 23: return np.rot90(np.rot90(m, 3, (0, 1)), 1, (2, 1))
    return m


# Inverse rotations
def grid_irot90(m, i=None):
    if i is None: i = np.random.randint(24)
    if i == 0 : return m
    if i == 1 : return np.rot90(m,                      1, (2, 0))
    if i == 2 : return np.rot90(m,                      2, (2, 0))
    if i == 3 : return np.rot90(m,                      3, (2, 0))
    if i == 4 : return np.rot90(m,                      1, (2, 1))
    if i == 5 : return np.rot90(m,                      1, (1, 2))
    if i == 6 : return          np.rot90(m, 1, (1, 0))
    if i == 7 : return np.rot90(np.rot90(m, 1, (2, 0)), 1, (1, 0))
    if i == 8 : return np.rot90(np.rot90(m, 2, (2, 0)), 1, (1, 0))
    if i == 9 : return np.rot90(np.rot90(m, 3, (2, 0)), 1, (1, 0))
    if i == 10: return np.rot90(np.rot90(m, 1, (2, 1)), 1, (1, 0))
    if i == 11: return np.rot90(np.rot90(m, 1, (1, 2)), 1, (1, 0))
    if i == 12: return          np.rot90(m, 2, (1, 0))
    if i == 13: return np.rot90(np.rot90(m, 1, (2, 0)), 2, (1, 0))
    if i == 14: return np.rot90(np.rot90(m, 2, (2, 0)), 2, (1, 0))
    if i == 15: return np.rot90(np.rot90(m, 3, (2, 0)), 2, (1, 0))
    if i == 16: return np.rot90(np.rot90(m, 1, (2, 1)), 2, (1, 0))
    if i == 17: return np.rot90(np.rot90(m, 1, (1, 2)), 2, (1, 0))
    if i == 18: return          np.rot90(m, 3, (1, 0))
    if i == 19: return np.rot90(np.rot90(m, 1, (2, 0)), 3, (1, 0))
    if i == 20: return np.rot90(np.rot90(m, 2, (2, 0)), 3, (1, 0))
    if i == 21: return np.rot90(np.rot90(m, 3, (2, 0)), 3, (1, 0))
    if i == 22: return np.rot90(np.rot90(m, 1, (2, 1)), 3, (1, 0))
    if i == 23: return np.rot90(np.rot90(m, 1, (1, 2)), 3, (1, 0))
    return m


def bgrid_rot90(m, i=None):
    if i is None: i = np.random.randint(24)
    if i == 0 : return m
    if i == 1 : return np.rot90(m,                      1, (1, 3))
    if i == 2 : return np.rot90(m,                      2, (1, 3))
    if i == 3 : return np.rot90(m,                      3, (1, 3))
    if i == 4 : return np.rot90(m,                      1, (2, 3))
    if i == 5 : return np.rot90(m,                      1, (3, 2))
    if i == 6 : return          np.rot90(m, 1, (1, 2))
    if i == 7 : return np.rot90(np.rot90(m, 1, (1, 2)), 1, (1, 3))
    if i == 8 : return np.rot90(np.rot90(m, 1, (1, 2)), 2, (1, 3))
    if i == 9 : return np.rot90(np.rot90(m, 1, (1, 2)), 3, (1, 3))
    if i == 10: return np.rot90(np.rot90(m, 1, (1, 2)), 1, (2, 3))

    if i == 11: return np.rot90(np.rot90(m, 1, (1, 2)), 1, (3, 2))
    if i == 12: return          np.rot90(m, 2, (1, 2))
    if i == 13: return np.rot90(np.rot90(m, 2, (1, 2)), 1, (1, 3))
    if i == 14: return np.rot90(np.rot90(m, 2, (1, 2)), 2, (1, 3))
    if i == 15: return np.rot90(np.rot90(m, 2, (1, 2)), 3, (1, 3))
    if i == 16: return np.rot90(np.rot90(m, 2, (1, 2)), 1, (2, 3))
    if i == 17: return np.rot90(np.rot90(m, 2, (1, 2)), 1, (3, 2))
    if i == 18: return          np.rot90(m, 3, (1, 2))
    if i == 19: return np.rot90(np.rot90(m, 3, (1, 2)), 1, (1, 3))
    if i == 20: return np.rot90(np.rot90(m, 3, (1, 2)), 2, (1, 3))
    if i == 21: return np.rot90(np.rot90(m, 3, (1, 2)), 3, (1, 3))
    if i == 22: return np.rot90(np.rot90(m, 3, (1, 2)), 1, (2, 3))
    if i == 23: return np.rot90(np.rot90(m, 3, (1, 2)), 1, (3, 2))
    return m


# Inverse rotations
def bgrid_irot90(m, i=None):
    if i is None: i = np.random.randint(24)
    if i == 0 : return m
    if i == 1 : return np.rot90(m,                      1, (3, 1))
    if i == 2 : return np.rot90(m,                      2, (3, 1))
    if i == 3 : return np.rot90(m,                      3, (3, 1))
    if i == 4 : return np.rot90(m,                      1, (3, 2))
    if i == 5 : return np.rot90(m,                      1, (2, 3))
    if i == 6 : return          np.rot90(m, 1, (2, 1))
    if i == 7 : return np.rot90(np.rot90(m, 1, (3, 1)), 1, (2, 1))
    if i == 8 : return np.rot90(np.rot90(m, 2, (3, 1)), 1, (2, 1))
    if i == 9 : return np.rot90(np.rot90(m, 3, (3, 1)), 1, (2, 1))
    if i == 10: return np.rot90(np.rot90(m, 1, (3, 2)), 1, (2, 1))
    if i == 11: return np.rot90(np.rot90(m, 1, (2, 3)), 1, (2, 1))
    if i == 12: return          np.rot90(m, 2, (2, 1))
    if i == 13: return np.rot90(np.rot90(m, 1, (3, 1)), 2, (2, 1))
    if i == 14: return np.rot90(np.rot90(m, 2, (3, 1)), 2, (2, 1))
    if i == 15: return np.rot90(np.rot90(m, 3, (3, 1)), 2, (2, 1))
    if i == 16: return np.rot90(np.rot90(m, 1, (3, 2)), 2, (2, 1))
    if i == 17: return np.rot90(np.rot90(m, 1, (2, 3)), 2, (2, 1))
    if i == 18: return          np.rot90(m, 3, (2, 1))
    if i == 19: return np.rot90(np.rot90(m, 1, (3, 1)), 3, (2, 1))
    if i == 20: return np.rot90(np.rot90(m, 2, (3, 1)), 3, (2, 1))
    if i == 21: return np.rot90(np.rot90(m, 3, (3, 1)), 3, (2, 1))
    if i == 22: return np.rot90(np.rot90(m, 1, (3, 2)), 3, (2, 1))
    if i == 23: return np.rot90(np.rot90(m, 1, (2, 3)), 3, (2, 1))
    return m


def grid_rot90s(m):
    rots = []

    rots.append(m)  # 0
    rots.append(np.rot90(rots[0], 1, (0, 2)))  # 1
    rots.append(np.rot90(rots[1], 1, (0, 2)))  # 2
    rots.append(np.rot90(rots[2], 1, (0, 2)))  # 3
    rots.append(np.rot90(rots[0], 1, (1, 2)))  # 4
    rots.append(np.rot90(rots[0], 1, (2, 1)))  # 5

    rots.append(np.rot90(rots[0], 1, (0, 1)))  # 6
    rots.append(np.rot90(rots[6], 1, (0, 2)))  # 7
    rots.append(np.rot90(rots[7], 1, (0, 2)))  # 8
    rots.append(np.rot90(rots[8], 1, (0, 2)))  # 9
    rots.append(np.rot90(rots[6], 1, (1, 2)))  # 10
    rots.append(np.rot90(rots[6], 1, (2, 1)))  # 11

    rots.append(np.rot90(rots[6], 1, (0, 1)))  # 12
    rots.append(np.rot90(rots[12], 1, (0, 2)))  # 13
    rots.append(np.rot90(rots[13], 1, (0, 2)))  # 14
    rots.append(np.rot90(rots[14], 1, (0, 2)))  # 15
    rots.append(np.rot90(rots[12], 1, (1, 2)))  # 16
    rots.append(np.rot90(rots[12], 1, (2, 1)))  # 17

    rots.append(np.rot90(rots[12], 1, (0, 1)))  # 18
    rots.append(np.rot90(rots[18], 1, (0, 2)))  # 19
    rots.append(np.rot90(rots[19], 1, (0, 2)))  # 20
    rots.append(np.rot90(rots[20], 1, (0, 2)))  # 21
    rots.append(np.rot90(rots[18], 1, (1, 2)))  # 22
    rots.append(np.rot90(rots[18], 1, (2, 1)))  # 23

    return rots


def torch_rot90(m, i=None):
    if i is None: i = np.random.randint(24)
    if i == 0 : return m
    if i == 1 : return torch.rot90(m,                         1, (1, 3))
    if i == 2 : return torch.rot90(m,                         2, (1, 3))
    if i == 3 : return torch.rot90(m,                         3, (1, 3))
    if i == 4 : return torch.rot90(m,                         1, (2, 3))
    if i == 5 : return torch.rot90(m,                         1, (3, 2))
    if i == 6 : return             torch.rot90(m, 1, (1, 2))
    if i == 7 : return torch.rot90(torch.rot90(m, 1, (1, 2)), 1, (1, 3))
    if i == 8 : return torch.rot90(torch.rot90(m, 1, (1, 2)), 2, (1, 3))
    if i == 9 : return torch.rot90(torch.rot90(m, 1, (1, 2)), 3, (1, 3))
    if i == 10: return torch.rot90(torch.rot90(m, 1, (1, 2)), 1, (2, 3))

    if i == 11: return torch.rot90(torch.rot90(m, 1, (1, 2)), 1, (3, 2))
    if i == 12: return             torch.rot90(m, 2, (1, 2))
    if i == 13: return torch.rot90(torch.rot90(m, 2, (1, 2)), 1, (1, 3))
    if i == 14: return torch.rot90(torch.rot90(m, 2, (1, 2)), 2, (1, 3))
    if i == 15: return torch.rot90(torch.rot90(m, 2, (1, 2)), 3, (1, 3))
    if i == 16: return torch.rot90(torch.rot90(m, 2, (1, 2)), 1, (2, 3))
    if i == 17: return torch.rot90(torch.rot90(m, 2, (1, 2)), 1, (3, 2))
    if i == 18: return             torch.rot90(m, 3, (1, 2))
    if i == 19: return torch.rot90(torch.rot90(m, 3, (1, 2)), 1, (1, 3))
    if i == 20: return torch.rot90(torch.rot90(m, 3, (1, 2)), 2, (1, 3))
    if i == 21: return torch.rot90(torch.rot90(m, 3, (1, 2)), 3, (1, 3))
    if i == 22: return torch.rot90(torch.rot90(m, 3, (1, 2)), 1, (2, 3))
    if i == 23: return torch.rot90(torch.rot90(m, 3, (1, 2)), 1, (3, 2))
    return m


# Inverse rotations
def torch_irot90(m, i=None):
    if i is None: i = np.random.randint(24)
    if i == 0 : return m
    if i == 1 : return torch.rot90(m,                         1, (3, 1))
    if i == 2 : return torch.rot90(m,                         2, (3, 1))
    if i == 3 : return torch.rot90(m,                         3, (3, 1))
    if i == 4 : return torch.rot90(m,                         1, (3, 2))
    if i == 5 : return torch.rot90(m,                         1, (2, 3))
    if i == 6 : return             torch.rot90(m, 1, (2, 1))
    if i == 7 : return torch.rot90(torch.rot90(m, 1, (3, 1)), 1, (2, 1))
    if i == 8 : return torch.rot90(torch.rot90(m, 2, (3, 1)), 1, (2, 1))
    if i == 9 : return torch.rot90(torch.rot90(m, 3, (3, 1)), 1, (2, 1))
    if i == 10: return torch.rot90(torch.rot90(m, 1, (3, 2)), 1, (2, 1))
    if i == 11: return torch.rot90(torch.rot90(m, 1, (2, 3)), 1, (2, 1))
    if i == 12: return             torch.rot90(m, 2, (2, 1))
    if i == 13: return torch.rot90(torch.rot90(m, 1, (3, 1)), 2, (2, 1))
    if i == 14: return torch.rot90(torch.rot90(m, 2, (3, 1)), 2, (2, 1))
    if i == 15: return torch.rot90(torch.rot90(m, 3, (3, 1)), 2, (2, 1))
    if i == 16: return torch.rot90(torch.rot90(m, 1, (3, 2)), 2, (2, 1))
    if i == 17: return torch.rot90(torch.rot90(m, 1, (2, 3)), 2, (2, 1))
    if i == 18: return             torch.rot90(m, 3, (2, 1))
    if i == 19: return torch.rot90(torch.rot90(m, 1, (3, 1)), 3, (2, 1))
    if i == 20: return torch.rot90(torch.rot90(m, 2, (3, 1)), 3, (2, 1))
    if i == 21: return torch.rot90(torch.rot90(m, 3, (3, 1)), 3, (2, 1))
    if i == 22: return torch.rot90(torch.rot90(m, 1, (3, 2)), 3, (2, 1))
    if i == 23: return torch.rot90(torch.rot90(m, 1, (2, 3)), 3, (2, 1))
    return m


def torch_rot90s(m):
    rots = torch.zeros([24]+list(m.shape), dtype=m.type()).to(m.device)
    rots[ 0] = m
    rots[ 1] = torch.rot90(m,                         1, (3, 1))
    rots[ 2] = torch.rot90(m,                         2, (3, 1))
    rots[ 3] = torch.rot90(m,                         3, (3, 1))
    rots[ 4] = torch.rot90(m,                         1, (3, 2))
    rots[ 5] = torch.rot90(m,                         1, (2, 3))
    rots[ 6] =             torch.rot90(m, 1, (2, 1))
    rots[ 7] = torch.rot90(torch.rot90(m, 1, (3, 1)), 1, (2, 1))
    rots[ 8] = torch.rot90(torch.rot90(m, 2, (3, 1)), 1, (2, 1))
    rots[ 9] = torch.rot90(torch.rot90(m, 3, (3, 1)), 1, (2, 1))
    rots[10] = torch.rot90(torch.rot90(m, 1, (3, 2)), 1, (2, 1))
    rots[11] = torch.rot90(torch.rot90(m, 1, (2, 3)), 1, (2, 1))
    rots[12] =             torch.rot90(m, 2, (2, 1))
    rots[13] = torch.rot90(torch.rot90(m, 1, (3, 1)), 2, (2, 1))
    rots[14] = torch.rot90(torch.rot90(m, 2, (3, 1)), 2, (2, 1))
    rots[15] = torch.rot90(torch.rot90(m, 3, (3, 1)), 2, (2, 1))
    rots[16] = torch.rot90(torch.rot90(m, 1, (3, 2)), 2, (2, 1))
    rots[17] = torch.rot90(torch.rot90(m, 1, (2, 3)), 2, (2, 1))
    rots[18] =             torch.rot90(m, 3, (2, 1))
    rots[19] = torch.rot90(torch.rot90(m, 1, (3, 1)), 3, (2, 1))
    rots[20] = torch.rot90(torch.rot90(m, 2, (3, 1)), 3, (2, 1))
    rots[21] = torch.rot90(torch.rot90(m, 3, (3, 1)), 3, (2, 1))
    rots[22] = torch.rot90(torch.rot90(m, 1, (3, 2)), 3, (2, 1))
    rots[23] = torch.rot90(torch.rot90(m, 1, (2, 3)), 3, (2, 1))

    return rots


def v_rot(v_, axes):
    assert len(v_) == 3
    axes = tuple(axes)
    v = np.asanyarray(copy.copy(v_))

    a = np.arange(0, 3)
    (a[axes[0]], a[axes[1]]) = (a[axes[1]], a[axes[0]])

    v[axes[1]] = -v[axes[1]]
    return np.array([v[a[0]], v[a[1]], v[a[2]]])


def vector_rot90s(v):
    rots = []

    rots.append(v)  # 0
    rots.append(v_rot(rots[0], (2, 0)))  # 1
    rots.append(v_rot(rots[1], (2, 0)))  # 2
    rots.append(v_rot(rots[2], (2, 0)))  # 3
    rots.append(v_rot(rots[0], (1, 0)))  # 4
    rots.append(v_rot(rots[0], (0, 1)))  # 5

    rots.append(v_rot(rots[0], (2, 1)))  # 6
    rots.append(v_rot(rots[6], (2, 0)))  # 7
    rots.append(v_rot(rots[7], (2, 0)))  # 8
    rots.append(v_rot(rots[8], (2, 0)))  # 9
    rots.append(v_rot(rots[6], (1, 0)))  # 10
    rots.append(v_rot(rots[6], (0, 1)))  # 11

    rots.append(v_rot(rots[6], (2, 1)))  # 12
    rots.append(v_rot(rots[12], (2, 0)))  # 13
    rots.append(v_rot(rots[13], (2, 0)))  # 14
    rots.append(v_rot(rots[14], (2, 0)))  # 15
    rots.append(v_rot(rots[12], (1, 0)))  # 16
    rots.append(v_rot(rots[12], (0, 1)))  # 17

    rots.append(v_rot(rots[12], (2, 1)))  # 18
    rots.append(v_rot(rots[18], (2, 0)))  # 19
    rots.append(v_rot(rots[19], (2, 0)))  # 20
    rots.append(v_rot(rots[20], (2, 0)))  # 21
    rots.append(v_rot(rots[18], (1, 0)))  # 22
    rots.append(v_rot(rots[18], (0, 1)))  # 23

    return rots


def vector_rot90(v, i=None):
    if i is None: i = np.random.randint(0, 24)
    vs = vector_rot90s(v)
    return vs[i]


def rotate_2d(V, rotation_matrix, inverse=False):
    batch_size = rotation_matrix.shape[0]
    box_size = V.shape[-1]
    device = V.device
    if inverse:
        rotation_matrix = rotation_matrix.transpose(1,2)
    rotation_matrix = torch.cat([rotation_matrix,torch.zeros([batch_size,2,1]).to(device)],dim=2)
    G = torch.nn.functional.affine_grid(
        rotation_matrix,
        (batch_size, 1, box_size,box_size),
        align_corners=False
    ).float()
    VV = torch.nn.functional.grid_sample(input=V, grid=G.to(device), mode='bilinear',  align_corners=False)
    return VV

if __name__ == "__main__":
    b = 61
    b2 = b/2-0.5
    vol = np.zeros((2, b, b, b))

    vol[0, 20, 10, 40] = 1
    vol[0, 12, 32, 11] = 3
    vol[0, 22, 12, 41] = 5
    vol[1, 22, 33, 41] = 2
    vol[1, 55, 33, 41] = 4
    vol[1, 58, 22,  4] = 4

    for irot in range(25):
        vol1 = bgrid_rot90(vol, irot)
        vol2 = bgrid_irot90(vol1, irot)

        if np.all(np.equal(vol, vol2)):
            print(irot, ", good!")
        else:
            print(irot, ", bad!")


