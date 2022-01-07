import numpy as np
import os

def read_obj(filename):
    """
        vert, face
    """


    faces = []
    vertices = []
    fid = open(filename, "r")
    node_counter = 0
    while True:

        line = fid.readline()
        if line == "":
            break
        while line.endswith("\\"):
            # Remove backslash and concatenate with next line
            line = line[:-1] + fid.readline()
        if line.startswith("v"):
            coord = line.split()
            coord.pop(0)
            node_counter += 1
            vertices.append(np.array([float(c) for c in coord]))

        elif line.startswith("f "):
            fields = line.split()
            fields.pop(0)

            # in some obj faces are defined as -70//-70 -69//-69 -62//-62
            cleaned_fields = []
            for f in fields:
                f = int(f.split("/")[0]) - 1
                if f < 0:
                    f = node_counter + f
                cleaned_fields.append(f)
            faces.append(np.array(cleaned_fields))
    fid.close()
    faces_np = np.row_stack(faces)
    vertices_np = np.row_stack(vertices)

    return vertices_np, faces_np


def read_obj_sample(filename):
    vert, face = read_obj(filename)
    down_file_name = '0.txt'  
    down = np.loadtxt(down_file_name).astype(np.int)
    idx_1723 = down[:, 1]
    return vert[idx_1723], face

def write_to_obj(filename, vertices, faces=None):
    if not filename.endswith('obj'):
        filename += '.obj'
    name = filename.split('/')[-1]
    path = filename.strip(name)
    if not os.path.exists(path):
        os.makedirs(path)
    num = vertices.shape[0]
    if faces is None:
        faces = np.loadtxt('/test/flownet3d/{:d}face.txt'.format(num), dtype=np.int)
    num_face = faces.shape[0]
    with open(filename, 'w') as f:
        for i in range(num):
            f.write('v {:f} {:f} {:f}\n'.format(*vertices[i].tolist()))
        for j in range(num_face):
            f.write('f {:d} {:d} {:d}\n'.format(*faces[j].tolist()))

def cal_rotation_matrix(rotation_angle=0, axis='x'):
    cos_value = np.cos(rotation_angle)
    sin_value = np.sin(rotation_angle)
    if axis == 'x':
        rotation_matrix = np.array(
            [
                [1., 0., 0.],
                [0., cos_value, -1*sin_value],
                [0., 1*sin_value, cos_value]
            ]
        )
    elif axis == 'y':
        rotation_matrix = np.array(
            [
                [cos_value, 0., sin_value],
                [0., 1., 0.],
                [-1*sin_value, 0., cos_value]
            ]
    )
    elif axis == 'z':
        rotation_matrix = np.array(
            [
                [cos_value, -1*sin_value, 0],
                [1*sin_value, cos_value, 0.],
                [0., 0., 1.]
            ]
        )

    else:
        print('axis input should in [\'x\', \'y\', \'z\']')

    return rotation_matrix

if __name__ == "__main__":
    # vertices = np.zeros([1723, 3])
    # write_to_obj('./test/tets.obj', vertices)
    # ver, face = read_obj('../flownet3d/template.obj')
    # transform = np.mean(ver, axis=0, keepdims=True)
    # print(transform)

    # ver -= transform
    # ro_x = cal_rotation_matrix(np.pi/2, 'x')
    # ro_z = cal_rotation_matrix(-np.pi/2, 'z')

    # ver = np.matmul(ver, ro_x)
    # ver = np.matmul(ver, ro_z)

    # write_to_obj('/test/zhenghuayu/pointtosmpl_100_60160_cp/results/test/template_6890_ro.obj', ver)
    from scipy.io import savemat

    vert, face = read_obj('../GeomFmaps/data/sample_faust/obj/template.obj')
    face += 1
    d = {'VERT': vert, 'TRIV':face.astype('float'), 'm':13776, 'n':6890}
    savemat('template.mat', d)




