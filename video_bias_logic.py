import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

def compute_video_bias_scores_from_upload(frame_path, embed_path, gender_path):
    try:
        # Load inputs
        frame_df = pd.read_csv(frame_path)
        embed_df = pd.read_csv(embed_path)
        gender_df = pd.read_csv(gender_path)

        # === Size Bias ===
        frame_df["size_ratio"] = frame_df["bbox_area"] / frame_df["frame_area"]
        size_bias_df = frame_df.groupby("video_name")["size_ratio"].mean().reset_index(name="size_bias")
        size_bias_df = size_bias_df.merge(gender_df[["video_name", "gender"]], on="video_name")
        size_bias_df["size_bias"] *= size_bias_df["gender"].map({"male": 1, "female": -1})
        size_bias_df.drop(columns=["gender"], inplace=True)

        # === Centering Bias ===
        frame_df["frame_diag_half"] = 0.5 * np.sqrt(frame_df["frame_width"]**2 + frame_df["frame_height"]**2)
        frame_df["center_norm"] = 1 - (frame_df["distance_to_center"] / frame_df["frame_diag_half"])
        centering_bias_df = frame_df.groupby("video_name")["center_norm"].mean().reset_index(name="centering_bias")
        centering_bias_df = centering_bias_df.merge(gender_df[["video_name", "gender"]], on="video_name")
        centering_bias_df["centering_bias"] *= centering_bias_df["gender"].map({"male": 1, "female": -1})
        centering_bias_df.drop(columns=["gender"], inplace=True)

        # === Screen Time Bias ===
        screen_time_df = frame_df.groupby("video_name").size().reset_index(name="detected_frames")
        screen_time_df = screen_time_df.merge(gender_df[["video_name", "gender"]], on="video_name")
        total_frames = screen_time_df["detected_frames"].sum()
        screen_time_df["screen_time_bias"] = screen_time_df["detected_frames"] * screen_time_df["gender"].map({"male": 1, "female": -1})
        screen_time_df["screen_time_bias"] /= total_frames
        screen_time_bias_df = screen_time_df[["video_name", "screen_time_bias"]]

        # === Embedding Bias ===
        embed_cols = [c for c in embed_df.columns if c.startswith("embed_")]
        embed_full = embed_df.merge(gender_df[["video_name", "gender"]], on="video_name")
        norms = np.linalg.norm(embed_full[embed_cols].values, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embed_full[embed_cols] /= norms

        male_centroid = embed_full[embed_full["gender"] == "male"][embed_cols].mean().values.reshape(1, -1)
        female_centroid = embed_full[embed_full["gender"] == "female"][embed_cols].mean().values.reshape(1, -1)

        def compute_embedding_bias(row):
            vec = row[embed_cols].values.reshape(1, -1)
            return cosine_distances(vec, female_centroid)[0][0] - cosine_distances(vec, male_centroid)[0][0]

        embed_full["embedding_bias"] = embed_full.apply(compute_embedding_bias, axis=1)
        embedding_bias_df = embed_full[["video_name", "embedding_bias"]]

        # === Motion Bias ===
        pose_cols = [c for c in frame_df.columns if c.startswith("pose_")]
        motion_vectors = []

        for video, group in frame_df.groupby("video_name"):
            try:
                fw, fh = group.iloc[0][["frame_width", "frame_height"]]
                poses = group.sort_values("frame")[pose_cols].dropna().values.astype(float).reshape(-1, 2, len(pose_cols) // 2)
                poses[:, 0, :] /= fw
                poses[:, 1, :] /= fh
                flat_poses = poses.reshape(poses.shape[0], -1)
                if len(flat_poses) < 2:
                    continue
                motion = np.abs(np.diff(flat_poses, axis=0)).mean(axis=0)
                motion_vectors.append((video, motion))
            except:
                continue

        motion_df = pd.DataFrame(motion_vectors, columns=["video_name", "motion_vector"])
        motion_df = motion_df.merge(gender_df[["video_name", "gender"]], on="video_name")
        male_centroid = np.mean([v for v, g in zip(motion_df["motion_vector"], motion_df["gender"]) if g == "male"], axis=0)
        female_centroid = np.mean([v for v, g in zip(motion_df["motion_vector"], motion_df["gender"]) if g == "female"], axis=0)

        def compute_motion_bias(vec):
            return cosine_distances([vec], [female_centroid])[0][0] - cosine_distances([vec], [male_centroid])[0][0]

        motion_df["motion_bias"] = motion_df["motion_vector"].apply(compute_motion_bias)
        motion_bias_df = motion_df[["video_name", "motion_bias"]]

        # === Combine All Bias Metrics ===
        bias_df = size_bias_df \
            .merge(centering_bias_df, on="video_name") \
            .merge(screen_time_bias_df, on="video_name") \
            .merge(embedding_bias_df, on="video_name") \
            .merge(motion_bias_df, on="video_name") \
            .merge(gender_df[["video_name", "category"]], on="video_name", how="left")

        # Normalize and compute PCA
        components = ["size_bias", "centering_bias", "screen_time_bias", "embedding_bias", "motion_bias"]
        standardized = pd.DataFrame(StandardScaler().fit_transform(bias_df[components]), columns=[f"{c}_std" for c in components])
        bias_df = pd.concat([bias_df, standardized], axis=1)
        pca = PCA(n_components=1)
        bias_df["combined_bias_pca"] = pca.fit_transform(standardized)
        bias_df["combined_bias_pca_scaled"] = MinMaxScaler((-1, 1)).fit_transform(bias_df[["combined_bias_pca"]])

        # Prepare final summary
        summary = bias_df.groupby("category")["combined_bias_pca_scaled"].mean().reset_index()
        summary.rename(columns={"combined_bias_pca_scaled": "bias_score"}, inplace=True)
        summary["bias_direction"] = summary["bias_score"].apply(lambda x: "male" if x > 0.05 else "female" if x < -0.05 else "neutral")

        return summary[["category", "bias_score", "bias_direction"]].to_dict(orient="records")

    except Exception as e:
        return {"error": str(e)}
