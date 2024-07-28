import torch
import numpy as np
import os
from scipy.spatial import cKDTree
import trimesh
import faiss

# def can_form_circle(points, beta):
#     A, B, C = points[:, :, 0, :], points[:, :, 1, :], points[:, :, 2, :]

#     v_AB = B - A
#     v_AC = C - A

#     norm_A = torch.sum(torch.square(A), dim=-1)
#     norm_B = torch.sum(torch.square(B), dim=-1)
#     norm_C = torch.sum(torch.square(C), dim=-1)

#     v_normal = torch.cross(v_AB, v_AC)

#     i1 = torch.divide(norm_B - norm_A, 2)
#     i2 = torch.divide(norm_C - norm_A, 2)
#     i3 = (v_normal @ A[:, 0].unsqueeze(-1)).squeeze()

#     right = torch.cat((i1.unsqueeze(-1), i2.unsqueeze(-1), i3.unsqueeze(-1)), dim=-1).unsqueeze(-1)
#     left = torch.cat((v_AB.unsqueeze(2), v_AC.unsqueeze(2), v_normal.unsqueeze(2)), dim=2)
    
#     try:
#         circumcenter = torch.linalg.solve(left, right).squeeze()
#         radius = torch.norm(A - circumcenter, dim=-1)
#     except:
#         circumcenter = torch.tensor([]).cuda()
#         radius = torch.tensor([]).cuda()

#         for i in range(A.size(0)):
#             try:
#               circum = torch.linalg.solve(left[i], right[i]).squeeze()
#               radi = torch.norm(A[i] - circum, dim=-1)
#             except:
#               circum = torch.zeros((A.size(1), 3))
#               radi = torch.zeros(A.size(1))
            
#             circumcenter = torch.cat((circumcenter, circum.unsqueeze(0).cuda()))
#             radius = torch.cat((radius, radi.unsqueeze(0).cuda()))

#     mask = torch.isfinite(radius) & (radius > 0)
#     condition = radius >= beta.unsqueeze(1)
#     mask = mask & condition

#     return mask, radius, circumcenter

def can_form_circle(points, beta):
    A, B, C = points[:, :, 0], points[:, :, 1], points[:, :, 2]
    
    v_AB = B - A
    v_AC = C - A
    
    norm_A = torch.sum(A**2, dim=-1)
    norm_B = torch.sum(B**2, dim=-1)
    norm_C = torch.sum(C**2, dim=-1)
    
    v_normal = torch.cross(v_AB, v_AC)
    
    i1 = (norm_B - norm_A) / 2
    i2 = (norm_C - norm_A) / 2
    i3 = torch.sum(v_normal * A, dim=-1)
    
    right = torch.stack((i1, i2, i3), dim=-1).unsqueeze(-1)
    left = torch.stack((v_AB, v_AC, v_normal), dim=-2)
    
    circumcenter = torch.zeros_like(A)
    radius = torch.zeros(A.shape[:-1], device=A.device)
    
    valid_indices = torch.arange(A.shape[0], device=A.device)
    
    while valid_indices.numel() > 0:
        try:
            solution = torch.linalg.solve(left[valid_indices], right[valid_indices])
            circumcenter[valid_indices] = solution.squeeze(-1)
            radius[valid_indices] = torch.norm(A[valid_indices] - circumcenter[valid_indices], dim=-1)
            break
        except RuntimeError:
            # If solve fails, process one by one
            for i in valid_indices:
                try:
                    sol = torch.linalg.solve(left[i], right[i])
                    circumcenter[i] = sol.squeeze()
                    radius[i] = torch.norm(A[i] - circumcenter[i], dim=-1)
                except RuntimeError:
                    # If individual solve fails, leave as zeros
                    pass
            break
    
    mask = torch.isfinite(radius) & (radius > 0) & (radius >= beta.unsqueeze(1))
    
    return mask, radius, circumcenter

def check_boundary(points, point_cloud, beta, batch_size=60):
    n = points.size(0)
    result_mask = torch.zeros((n,), dtype=torch.bool)

    num = n//batch_size
    points = points.view(-1, num, points.size(1), points.size(2), points.size(3))
    beta = beta.view(-1, num)

    for i in range(batch_size):
        current_points = points[i]

        boundary_mask, radius, circumcenter = can_form_circle(current_points, beta[i])

        for j, line in enumerate(boundary_mask):
            if line.any():
                distances = torch.norm(point_cloud.unsqueeze(0) - circumcenter[j].unsqueeze(1), dim=2)
                num_points_within_sphere = (distances <= radius[j].unsqueeze(1)).sum(dim=1)

                line &= (num_points_within_sphere == 3)

                if line.any():
                    result_mask[i*num + j] = True

    return result_mask

def calculate_direction_vector(cloud, k=8):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    point_cloud = cloud.to(device)

    n, d = cloud.shape
    nlist = 2 ** max(0, int(2*(np.log10(n)-1)))

    if (n < k + 1):
      return torch.zeros(n, d), torch.zeros(n, d)

    np_cloud = point_cloud.cpu().data.numpy()

    quantizer = faiss.IndexFlatL2(d)

    index = faiss.IndexIVFPQ(quantizer, d, nlist, 3, 8)
    train_points = np_cloud[np.random.choice(n, max(256, n), replace=(n < 256))]
    index.train(train_points)
    index.add(np_cloud)

    _, indices = index.search(np_cloud, k+1)
    indices = torch.from_numpy(indices).to(device)
    indices = indices[:, 1:].reshape(-1)

    neighbor_points = torch.index_select(point_cloud, dim=0, index=indices)
    neighbor_points = neighbor_points.reshape(n, k, 3)
    centroid_points = torch.mean(neighbor_points, dim=1, keepdim=True)

    neighbor_vectors = neighbor_points - point_cloud.unsqueeze(1)
    centroid_vectors = neighbor_points - centroid_points

    covariances = (1 / k) * torch.matmul(centroid_vectors.transpose(1, 2), centroid_vectors)

    _, eigvecs = torch.linalg.eigh(covariances.cpu())
    normals = eigvecs[:, :, 0].cuda()

    plane_normal = torch.cross(normals.unsqueeze(1).repeat(1, k, 1), neighbor_vectors, dim=2)

    perpendicular_vectors = torch.mean(plane_normal, dim=1)

    combi_idx = torch.combinations(torch.arange(k))
    combi_idx = combi_idx.unsqueeze(0).expand(n, -1, -1)
    combi_idx = combi_idx.cuda()

    reshaped_neighbor = neighbor_points.unsqueeze(1).expand(-1, combi_idx.size(1), -1, -1)
    
    # Gather the indices along the second dimension
    neighbor_combi = reshaped_neighbor.gather(2, combi_idx.unsqueeze(-1).expand(-1, -1, -1, 3))
    pc_expanded = point_cloud.unsqueeze(1).unsqueeze(2).expand(-1, neighbor_combi.size(1), 1, -1)
    points = torch.cat((pc_expanded, neighbor_combi), dim=2)

    neighbor_points = neighbor_points.view(-1, k, 1, 3)

    # Compute pairwise distances using broadcasting
    neighbor_distances = torch.norm(neighbor_points - neighbor_points.permute(0, 2, 1, 3), dim=-1)

    # Set the diagonal distances to infinity to exclude self-distances
    torch.diagonal(neighbor_distances, dim1=-2, dim2=-1).fill_(float('inf'))

    min_vals, _ = torch.min(neighbor_distances, dim=-1)

    mean = torch.mean(min_vals, dim=1)
    std = torch.std(min_vals, dim=1)
    beta = mean + 2 * std

    mask = check_boundary(points, point_cloud, beta)

    torch.cuda.empty_cache()
    return point_cloud[mask], perpendicular_vectors[mask]


def search_nearest_point(point_batch, point_gt):
    # num_point_batch = number of point clouds // 60
    # num_point_gt = number of point clouds
    # point_batch is sample points
    # point_gt is raw points

    num_point_batch, num_point_gt = point_batch.shape[0], point_gt.shape[0]
    
    # point_batch and point_gt has the same shape 
    # [num_point_batch, num_point_gt, 3]
    point_batch = point_batch.unsqueeze(1).repeat(1, num_point_gt, 1)
    point_gt = point_gt.unsqueeze(0).repeat(num_point_batch, 1, 1)
    
    # distances has shape [num_point_batch, num_point_gt]
    # dis_idx returns the index of nearest ground truth points to 
    # each of the sample points
    
    distances = torch.sqrt(torch.sum((point_batch-point_gt) ** 2, axis=-1) + 1e-12) 
    dis_idx = torch.argmin(distances, axis=1).detach().cpu().numpy()

    torch.cuda.empty_cache()
    return dis_idx

def write_ply(points, color, filename, mode="w"):
    with open(filename, mode) as f:
        # Write the header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(len(points)))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # Write the points with the same color
        for point in points:
            f.write("{} {} {} {} {} {}\n".format(point[0], point[1], point[2], color[0], color[1], color[2]))

def calculate_normal_vectors(nearest_neighbors):
    # Extract coordinates of points and neighbors
    p1, p2, p3 = nearest_neighbors[:, 0][:, None], nearest_neighbors[:, 1][:, None], nearest_neighbors[:, 2][:, None]
    # Calculate vectors representing two edges of the triangles
    edge1 = p2 - p1
    edge2 = p3 - p1
    
    # Calculate the cross product of the two edges
    normal_vectors = np.cross(edge1, edge2, axis=2)
    
    # Normalize the normal vectors
    normal_vectors /= np.linalg.norm(normal_vectors, axis=2)[:, :, np.newaxis]
    
    return normal_vectors.squeeze(1)

def process_data(data_dir, dataname):
    print(data_dir, dataname)
    if os.path.exists(os.path.join(data_dir, 'input', dataname) + '.ply'):
        pointcloud = trimesh.load(os.path.join(data_dir, 'input', dataname) + '.ply').vertices
        pointcloud = np.asarray(pointcloud)
    elif os.path.exists(os.path.join(data_dir, 'input', dataname) + '.xyz'):
        pointcloud = np.loadtxt(os.path.join(data_dir, 'input', dataname) + '.xyz')
    elif os.path.exists(os.path.join(data_dir, 'input', dataname) + '.npy'):
        pointcloud = np.load(os.path.join(data_dir, 'input', dataname) + '.npy')
    else:
        print('Only support .ply, .xyz or .npy data. Please adjust your data format.')
        exit()
    shape_scale = np.max([np.max(pointcloud[:,0])-np.min(pointcloud[:,0]),np.max(pointcloud[:,1])-np.min(pointcloud[:,1]),np.max(pointcloud[:,2])-np.min(pointcloud[:,2])])
    shape_center = [(np.max(pointcloud[:,0])+np.min(pointcloud[:,0]))/2, (np.max(pointcloud[:,1])+np.min(pointcloud[:,1]))/2, (np.max(pointcloud[:,2])+np.min(pointcloud[:,2]))/2]
    pointcloud = pointcloud - shape_center
    pointcloud = pointcloud / shape_scale
    
    POINT_NUM = pointcloud.shape[0] // 60
    # POINT_NUM_GT is a nearest number to pointcloud.shape 
    # but divisible by 60
    POINT_NUM_GT = pointcloud.shape[0] // 60 * 60
    QUERY_EACH = 1000000//POINT_NUM_GT
    
    # print(f"Point num: {POINT_NUM} | Point num GT: {POINT_NUM_GT} | Query each: {QUERY_EACH}")

    point_idx = np.random.choice(pointcloud.shape[0], POINT_NUM_GT, replace = False)
    pointcloud = pointcloud[point_idx,:]
    ptree = cKDTree(pointcloud)
    
    sigmas = []
    normal = []
    
    for p in np.array_split(pointcloud,100,axis=0):
        d = ptree.query(p,51)
        sigmas.append(d[0][:,-1])
    
    # sigma has 100 elements
    sigmas = np.concatenate(sigmas)
    
    sample = []
    sample_near = []
    normal = []
    
    k = 8

    for i in range(QUERY_EACH):

        theta = 0.25
        scale = max(theta, theta * np.sqrt(POINT_NUM_GT / 20000))
        tt = pointcloud + scale*np.expand_dims(sigmas,-1) * np.random.normal(0.0, 1.0, size=pointcloud.shape)
        sample.append(tt)
        
        # tt is raw point cloud added noise as the sample point
        # tt has shape [60, POINT_NUM, 3]
        tt = tt.reshape(-1,POINT_NUM,3)
        
        sample_near_tmp = []
        normal_tmp = []
        
        # find the nearest ground truth point to each of sample points.
        for j in range(tt.shape[0]):
            d = ptree.query(tt[j], k)
            
            neighbor_points = pointcloud[d[1]]
            neighbor_points = neighbor_points.reshape(-1, k, 3)
            neighbor_vectors = neighbor_points - tt[j][:, np.newaxis, :]
            covariances = np.matmul(neighbor_vectors.transpose(0, 2, 1), neighbor_vectors)
            _, eigvecs = np.linalg.eigh(covariances)
            normal_vectors = eigvecs[:, :, 0]
            
            normal_vectors = normal_vectors / (np.linalg.norm(normal_vectors, axis=1))[:, np.newaxis]            
            normal_tmp.append(normal_vectors)
            
            nearest_idx = search_nearest_point(torch.tensor(tt[j]).float().cuda(), torch.tensor(pointcloud).float().cuda())
            nearest_points = pointcloud[nearest_idx]
            # near_pints has shape [POINT_NUM, 3]
            nearest_points = np.asarray(nearest_points).reshape(-1,3)
            sample_near_tmp.append(nearest_points)
        
        sample_near_tmp = np.asarray(sample_near_tmp)
        normal_tmp = np.asarray(normal_tmp)
        # sample_near_tmp has shape [POINT_NUM * 60, 3] or [POINT_NUM_GT, 3]
        sample_near_tmp = sample_near_tmp.reshape(-1,3)
        normal_tmp = normal_tmp.reshape(-1,3)
        
        sample_near.append(sample_near_tmp)
        normal.append(normal_tmp)
    
    sample = np.asarray(sample)
    sample_near = np.asarray(sample_near)
    normal = np.asarray(normal)

    # save sample_near and sample points to query_data directory with .npz format
    os.makedirs(os.path.join(data_dir, 'query_data'), exist_ok=True)
    np.savez(os.path.join(data_dir, 'query_data', dataname)+'.npz', sample = sample, point = pointcloud, sample_near = sample_near, normal = normal)
    

class Dataset:
    def __init__(self, conf, dataname):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        self.data_name = dataname + '.npz'
        self.is_optimized = False
        

        if os.path.exists(os.path.join(self.data_dir, 'query_data', self.data_name)):
            print('Query data existing. Loading data...')
        else:
            print('Query data not found. Processing data...')
            process_data(self.data_dir, dataname)

        load_data = np.load(os.path.join(self.data_dir, 'query_data', self.data_name))
        
        self.point = np.asarray(load_data['sample_near']).reshape(-1,3)
        self.sample = np.asarray(load_data['sample']).reshape(-1,3)
        self.point_gt = np.asarray(load_data['point']).reshape(-1,3)
        self.point_gt_raw = np.asarray(load_data['point']).reshape(-1,3)
    
        # sample_points_num = QUERY_EACH - 1
        self.sample_points_num = self.sample.shape[0]-1

        self.object_bbox_min = np.array([np.min(self.point[:,0]), np.min(self.point[:,1]), np.min(self.point[:,2])]) -0.05
        self.object_bbox_max = np.array([np.max(self.point[:,0]), np.max(self.point[:,1]), np.max(self.point[:,2])]) +0.05

        self.point = torch.from_numpy(self.point).to(self.device).float()
        self.sample = torch.from_numpy(self.sample).to(self.device).float()
        self.point_gt_raw = torch.from_numpy(self.point_gt_raw).to(self.device).float()
        
        self.point_gt = torch.from_numpy(self.point_gt).to(self.device).float()
        
        print('NP Load data: End')

    def get_train_data(self, batch_size):
        index_coarse = np.random.choice(10, 1)
        index_fine = np.random.choice(self.sample_points_num//10, batch_size, replace = False)
        index = index_fine * 10 + index_coarse

        points = self.point[index]
        sample = self.sample[index]
        return points, sample, self.point_gt
    
    def gen_extra_points(self, iter_step, p=0.1, knn=8, alpha=3, extra_points = 240):
        num_point = self.point_gt.size(0) // 60 * 60
        # print(p, knn, alpha, extra_points)
        selected_gt, dir_vec = calculate_direction_vector(self.point_gt[:num_point], knn)
        
        dir_norm = torch.norm(dir_vec, dim=1)
        _, sorted_indices = torch.sort(dir_norm, descending=True)

        num = min(max(extra_points, int(dir_norm.size(0) * p) // 60 * 60), 2040)
        sorted_indices = sorted_indices[:num]

        if iter_step % 1000 == 0:
            write_ply(selected_gt[sorted_indices], (255, 255, 0), "{}_new_boundary.ply".format(iter_step))

        generated_point = selected_gt + alpha * dir_vec
        generated_point = generated_point[sorted_indices]

        torch.cuda.empty_cache()

        return generated_point.detach()
