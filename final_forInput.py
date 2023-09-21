import cv2
import numpy as np
import os
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg

def color_optimization(target_image, scribble_image, sigma_r):
    target_yuv = cv2.cvtColor(target_image, cv2.COLOR_BGR2YUV)
    target_gray = target_yuv[:,:,0].astype(np.float64)
    target_shape = target_gray.shape
    
    # 計算像素的4連通鄰居索引
    neighbor_indices = get_neighbor_indices(target_shape)

    # 構建稀疏矩陣
    A = build_sparse_matrix(target_shape, neighbor_indices, sigma_r)

    # 構建目標顏色矢量
    target_colors = target_yuv[:,:,1:].reshape(-1, 2).astype(np.float64)

    # 構建權重矩陣
    weights = build_weight_matrix(target_gray, target_shape, neighbor_indices, sigma_r)

    # 構建目標矢量
    target_vector = np.sum(weights * target_colors[neighbor_indices], axis=1)

    # 解線性系統
    result_vector = linalg.spsolve(A, target_vector)

    # 將結果重新塑造為圖像
    result_colors = result_vector.reshape(target_shape[0], target_shape[1], 2)
    result_yuv = np.zeros((target_shape[0], target_shape[1], 3), dtype=np.uint8)
    result_yuv[:,:,0] = target_yuv[:,:,0]  # 使用原始Y通道
    result_yuv[:,:,1:] = result_colors.astype(np.uint8)
    
    # 將YUV圖像轉換為BGR圖像
    result_image = cv2.cvtColor(result_yuv, cv2.COLOR_YUV2BGR)

    return result_image


def get_neighbor_indices(image_shape):
    height, width = image_shape
    
    indices = np.arange(height * width).reshape(height, width)
    neighbor_indices = np.zeros((height, width, 4), dtype=np.int32)

    neighbor_indices[:,:,0] = np.roll(indices, 1, axis=1)  # 左鄰居
    neighbor_indices[:,:,1] = np.roll(indices, -1, axis=1)  # 右鄰居
    neighbor_indices[:,:,2] = np.roll(indices, 1, axis=0)  # 上鄰居
    neighbor_indices[:,:,3] = np.roll(indices, -1, axis=0)  # 下鄰居

    return neighbor_indices


def build_sparse_matrix(image_shape, neighbor_indices, sigma_r):
    height, width = image_shape
    num_pixels = height * width
    num_neighbors = 4

    row_indices = np.repeat(np.arange(num_pixels), num_neighbors)
    col_indices = neighbor_indices.flatten()
    weights = np.exp(-np.square(image[row_indices//width, row_indices%width] - image[col_indices//width, col_indices%width]) / (2 * sigma_r**2))

    A = sparse.coo_matrix((weights, (row_indices, col_indices)), shape=(num_pixels, num_pixels)).tocsr()
    A += sparse.diags(-np.sum(A, axis=1).A.flatten(), 0)  # 將對角線元素設置為行總和的負值

    return A


def build_weight_matrix(image, image_shape, neighbor_indices, sigma_r):
    height, width = image_shape
    num_pixels = height * width
    num_neighbors = 4

    differences = np.square(image.flatten()[neighbor_indices] - image.reshape(-1, 1))
    weights = np.exp(-differences / (2 * sigma_r**2))

    return weights


def colorize_image(target_image, reference_images):
    # 步驟 1：使用 SIFT 找到匹配點
    sift = cv2.SIFT_create()
    keypoints_target, descriptors_target = sift.detectAndCompute(target_image, None)

    keypoints_references = []
    descriptors_references = []
    for reference_image in reference_images:
        keypoints, descriptors = sift.detectAndCompute(reference_image, None)
        keypoints_references.append(keypoints)
        descriptors_references.append(descriptors)

    # 步驟 2：使用 RANSAC 過濾不滿意的圖像
    match_threshold = 0.6
    # match_threshold = 0.4  # 调整 match_threshold 的值
    # inlier_threshold = 2.0  # 调整 inlier_threshold 的值
    inlier_threshold = 10.0

    good_matches = []
    for i in range(len(reference_images)):
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(descriptors_references[i], descriptors_target, k=2)
        good_matches_i = []
        for m, n in matches:
            if m.distance < match_threshold * n.distance:
                good_matches_i.append(m)
        good_matches.append(good_matches_i)

    # 使用 RANSAC 找到每個參考圖像的內點
    inliers = []
    homographies = []
    for i in range(len(reference_images)):
        if len(good_matches[i]) > 10:
            src_pts = np.float32([keypoints_references[i][m.queryIdx].pt for m in good_matches[i]]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints_target[m.trainIdx].pt for m in good_matches[i]]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, inlier_threshold)
            inliers.append(np.sum(mask) > 10)
            homographies.append(M)
        else:
            inliers.append(False)
            homographies.append(None)  # 添加 None 到列表中，以保持索引一致性

    # 步驟 3：計算光照和反射
    illumination = np.zeros(target_image.shape, dtype=np.float64)
    reflectance = np.zeros(target_image.shape, dtype=np.float64)
    for i in range(len(reference_images)):
        if inliers[i] and homographies[i] is not None:  # 確認 homographies[i] 不為 None
            # 對參考圖像應用投影變換
            projected_image = cv2.warpPerspective(reference_images[i], homographies[i], \
                                                  (target_image.shape[1], target_image.shape[0]))

            # 計算光照（使用最大值運算）
            weight = 0.1 # 調整權重以平衡光照估計和反射估計
            illumination = weight * illumination + (1 - weight) * projected_image.astype(np.float64)
            # illumination = np.maximum(illumination, projected_image.astype(np.float64))
            # illumination = (illumination + projected_image.astype(np.float64)) / 2


            # 計算反射（使用差值運算）
            # diff = cv2.absdiff(projected_image, target_image)
            # reflectance += diff.astype(np.float64)

            diff = np.abs(projected_image.astype(np.float64) - target_image.astype(np.float64))
            reflectance += diff



    # 步驟 4：恢復內在成分
    intrinsic_components = reflectance / (illumination + 1e-6)
    intrinsic_components[np.isnan(intrinsic_components)] = 0  # 將NaN值設置為0
    intrinsic_components[np.isinf(intrinsic_components)] = 0  # 將無窮大值設置為0

    # 步驟 5：填充缺失區域
    missing_mask = np.zeros(target_image.shape[:2], dtype=np.uint8)
    missing_mask[np.where(target_image.sum(axis=2) == 0)] = 255
    missing_mask = cv2.dilate(missing_mask, np.ones((15, 15), np.uint8), iterations=1)
    filled_image = cv2.inpaint(target_image, missing_mask, 3, cv2.INPAINT_NS)

    # 步驟 6：恢復結果
    result = filled_image.astype(np.float64) + intrinsic_components * (illumination - filled_image.astype(np.float64))
    result[np.isnan(result)] = filled_image[np.isnan(result)]  # 將NaN值設置為填充圖像對應位置的像素值
    result[np.isinf(result)] = filled_image[np.isinf(result)]  # 將無窮大值設置為填充圖像對應位置的像素值
    result[np.isnan(result)] = 0  # 將NaN值設置為0
    result[np.isinf(result)] = 0  # 將無窮大值設置為0
    result = np.clip(result, 0, 255)  # 將結果限制在0到255之間

    return result.astype(np.uint8)


# 載入目標圖像和參考圖像
target_image = cv2.imread("newtarget_gray.jpg")
reference_folder = "input"
# reference_folder = "input1"

reference_images = []
for filename in os.listdir(reference_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(reference_folder, filename)
        image = cv2.imread(image_path)
        reference_images.append(image)

# 進行圖像上色
colorized_image = colorize_image(target_image, reference_images)

# 保存結果
cv2.imwrite('colorized_image_finalForInput.jpg', colorized_image)
# cv2.imshow('colorized_image_finalForInput.jpg', colorized_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
