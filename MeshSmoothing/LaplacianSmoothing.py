from tqdm import tqdm
import numpy as np
import trimesh
import math
from scipy.linalg import sqrtm
from scipy import sparse
from scipy.sparse.linalg import spsolve

class LaplacianSmoothing():
    def __init__(self, h, iteration, mesh:trimesh, cell:str, mode:str) -> None:
        self.h = h   # h 代表欧拉公式中的步长
        self.iteration = iteration# iteration 代表迭代次数
        self.mesh = mesh # 三角形网格
        self.cell = cell # 局部平均区域的选取方式 'Barycentric cell' 'voronoi cell'
        self.mode = mode # 'uniform' 'cot'
    
    def ComputeTriangleArea_Barycentric(self, a, b, c):
        ab = b - a
        ac = c - a
        S = 0.5 * np.linalg.norm(np.cross(ab, ac))
        return S
    
    def ComputeTriangleArea_Voronoi(self, a, b, c):
        #     a
        #     /\
        #   b/__\c
        # a是靠近voronoi cell的中心
        area = 0
        S = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
        ab = np.linalg.norm(b - a)
        ac = np.linalg.norm(c - a)
        bc = np.linalg.norm(c - b)
        R = (ab * bc * ac) / (4 * S)
        cos = (ac * ac + bc * bc - ab * ab) / (2 * ac * bc)
        area += 0.5 * 0.5 * ab * R * cos
        cos = (ab * ab + bc * bc - ac * ac) / (2 * ab * bc)
        area += 0.5 * 0.5 * ac * R * cos
        return area
    
    def ComputeAngle(self, a, b, c):
        # 计算角A
        ab = b - a
        ac = c - a
        cosA = ab @ ac / (np.linalg.norm(ab) * np.linalg.norm(ac))
        A = np.arccos(cosA)
        return A
    
    def GetAdjacencyPoints(self, p1, p2):
        # 找到计算余弦拉普拉斯时的那两个角对应的点
        AdjacencyFaces = self.mesh.vertex_faces[p1][self.mesh.vertex_faces[p1] != -1]
        adjacency_pts = []
        for face in AdjacencyFaces:
            if p2 in self.mesh.faces[face]:
                pts = list(self.mesh.faces[face].copy())
                pts.remove(p1)
                pts.remove(p2)
                adjacency_pts.append(pts[0])
        return adjacency_pts

    def Compute_D(self):
        # 计算对角矩阵
        n = self.mesh.vertices.shape[0]
        D = []
        if self.mode == 'uniform':
            for i in range(n):
                D.append(len(self.mesh.vertex_neighbors[i]))
            D = np.diag(D)
        elif self.mode == 'cot':
            if self.cell == 'Barycentric':
                for i in range(n):
                    area = 0
                    AdjacencyFaces = self.mesh.vertex_faces[i][self.mesh.vertex_faces[i] != -1]
                    for face in AdjacencyFaces:
                        [a, b, c] = self.mesh.faces[face]
                        a = np.array(self.mesh.vertices[a])
                        b = np.array(self.mesh.vertices[b])
                        c = np.array(self.mesh.vertices[c])
                        area += self.ComputeTriangleArea_Barycentric(a, b, c)
                    D.append(area / 3)
                D = np.diag(D)
            elif self.cell == 'voronoi':
                for i in range(n):
                    area = 0
                    AdjacencyFaces = self.mesh.vertex_faces[i][self.mesh.vertex_faces[i] != -1]
                    for face in AdjacencyFaces:
                        pts = list(self.mesh.faces[face].copy())
                        pts.remove(i)
                        a = np.array(self.mesh.vertices[i])
                        b = np.array(self.mesh.vertices[pts[0]])
                        c = np.array(self.mesh.vertices[pts[1]])
                        area += self.ComputeTriangleArea_Voronoi(a, b, c)
                    D.append(area)
                D = np.diag(D)
        return D              
      
    def Compute_A(self):
        # 计算邻接矩阵
        n = self.mesh.vertices.shape[0]
        A = np.zeros((n, n))
        if self.mode == 'uniform':
            for i in range(n):
                for neighbor in self.mesh.vertex_neighbors[i]:
                    A[i, neighbor] = 1
                A[i, i] = -np.sum(A[i, :])
        elif self.mode == 'cot':
            for i in range(n):
                neighbors = self.mesh.vertex_neighbors[i]
                boundary = False
                for neighbor in neighbors:  
                    pts = self.GetAdjacencyPoints(i, neighbor)
                    if len(pts) == 0 or len(pts) == 1:
                        boundary = True
                    elif len(pts) == 2:
                        a = np.array(self.mesh.vertices[i])
                        b = np.array(self.mesh.vertices[neighbor])
                        c = np.array(self.mesh.vertices[pts[0]])
                        d = np.array(self.mesh.vertices[pts[1]])
                        alpha = self.ComputeAngle(c, a, b)
                        beta = self.ComputeAngle(d, a, b)
                        cot1 = np.clip(1 / math.tan(alpha), -10.0, 10.0)
                        cot2 = np.clip(1 / math.tan(beta), -10.0, 10.0)
                        A[i, neighbor] = 0.5 * (cot1 + cot2)
                A[i, i] = -np.sum(A[i, :])
                if boundary:
                    A[i, :] = 0
                    A[i, i] = 1
        return A

    def smooth(self):
        vertices = np.array(self.mesh.vertices)
        f_t = vertices.copy()
        if self.mode == 'uniform':
            D = self.Compute_D()
            print("Computing Matrix D...")
            A = self.Compute_A()
            print("Computing Matrix A...")
            L = sparse.csc_matrix(D - self.h * A)
            for i in tqdm(range(self.iteration)):
                f_t = spsolve(L, D @ f_t)

        else:
            for i in tqdm(range(self.iteration)):
            # print("{:d}/{:d} iteration".format(i + 1, self.iteration))
                D = self.Compute_D()
                print("Computing Matrix D...")
                A = self.Compute_A()
                print("Computing Matrix A...")
                # print(A)
                # print(D)
                L = sparse.csc_matrix(D - self.h * A)
                f_t = spsolve(L, D @ f_t)
        
        faces = np.array(self.mesh.faces)
        newMesh = trimesh.Trimesh(vertices=f_t, faces=faces)

        return newMesh


    
