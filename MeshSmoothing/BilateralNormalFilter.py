import trimesh
import numpy as np

class bilateral_normal_filter():

    def __init__(self,mesh:trimesh, sigma=0.5, normal_iter=20, vertex_iter=15):
        self.mesh = mesh
        self.sigma = sigma #法向量高斯函数的方差
        self.iter1 = normal_iter #法向量迭代次数
        self.iter2 = vertex_iter #顶点迭代次数
        self.face_normals = mesh.face_normals.copy() #面法向
        self.face_areas = mesh.area_faces #面积
        self.face_centers = mesh.triangles_center #面中心
        self.new_normals = None


    def face_map(self):
        # 寻找每个面的相邻面，为双边滤波做预处理
        adjacent_faces = [[] for _ in range(len(self.mesh.faces))]
        face_adjacency = self.mesh.face_adjacency
        for f0, f1 in face_adjacency:
            adjacent_faces[f0].append(f1)
            adjacent_faces[f1].append(f0)
        # print(len(adjacent_faces))
        return adjacent_faces

    def Compute_sigma(self, adjacent_faces):
        # 不直接硬编码重心高斯函数的方差，而是用相邻面之间中心点距离的平均值替代，更符合模型特征
        sigma_center = 0
        for i, neighbors in enumerate(adjacent_faces):
            for j in neighbors:
                sigma_center += np.linalg.norm(self.face_centers[i] - self.face_centers[j])
        sigma_center /= (len(self.mesh.faces) * 3)
        # print(sigma_center)
        return sigma_center

    def update_faces(self):
        adjacent_faces = self.face_map()
        sigma_center = self.Compute_sigma(adjacent_faces)
        self.new_normals = self.face_normals.copy()
        for _ in range(int(self.iter1)):
            updated_normals = np.zeros_like(self.new_normals)
            for i, neighbors in enumerate(adjacent_faces):
                n_i = self.new_normals[i]
                c_i = self.face_centers[i]
                Ki = 0
                sum_vec = np.zeros(3)
                for j in neighbors:
                    n_j = self.new_normals[j]
                    c_j = self.face_centers[j]
                    area_j = self.face_areas[j]

                    delta_center = np.linalg.norm(c_i - c_j)
                    delta_normal = np.linalg.norm(n_i - n_j)
                    W_s = np.exp(-delta_center ** 2 / (2 * sigma_center ** 2))
                    W_r = np.exp(-delta_normal ** 2 / (2 * self.sigma ** 2))

                    weight = area_j * W_s * W_r
                    sum_vec += weight * n_j
                    Ki += weight
                    # print(Ki)

                updated_normals[i] = sum_vec / Ki if Ki > 1e-6 else n_i # 除零就不更新
                updated_normals[i] /= np.linalg.norm(updated_normals[i])
            self.new_normals = updated_normals

    def update_vertex(self):
        vertices = self.mesh.vertices.copy()
        for _ in range(int(self.iter2)):
            new_vertices = vertices.copy()
            for vid in range(len(vertices)):
                adjacent_faces_idx = self.mesh.vertex_faces[vid]
                adjacent_faces_idx = adjacent_faces_idx[adjacent_faces_idx != -1]
                if len(adjacent_faces_idx) == 0:
                    continue
                x_i = vertices[vid]
                delta = np.zeros(3)
                for fid in adjacent_faces_idx:
                    c_j = self.face_centers[fid]
                    n_j = self.new_normals[fid]
                    proj = np.dot((c_j - x_i), n_j)
                    delta += proj * n_j
                new_vertices[vid] = x_i + delta / len(adjacent_faces_idx)
            vertices = new_vertices
        
        newmesh = self.mesh.copy()
        newmesh.vertices = vertices
        return newmesh