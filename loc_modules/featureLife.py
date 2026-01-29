"""
Feature Lifecycle Logger

Tracks features through the entire localization pipeline and logs
uncertainty/confidence signals at each stage.

Goal: Understand what makes features succeed by measuring uncertainty
      at every transition point.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class LifecycleStage(Enum):
    """Stages a feature passes through"""
    EXTRACTED = "extracted"
    QUANTIZED = "quantized"
    VOCAB_ASSIGNED = "vocab_assigned"
    MATCHED = "matched"
    RATIO_TESTED = "ratio_tested"
    RANSAC_SAMPLED = "ransac_sampled"
    RANSAC_INLIER = "ransac_inlier"
    POSE_CONTRIBUTOR = "pose_contributor"


class DeathCause(Enum):
    """Why a feature died"""
    ALIVE = "alive"
    WEAK_DETECTOR = "weak_detector"
    QUANTIZATION_LOSS = "quantization_loss"
    NO_VOCAB_MATCH = "no_vocab_match"
    NO_CANDIDATE = "no_candidate"
    RATIO_TEST_FAILED = "ratio_test_failed"
    RANSAC_OUTLIER = "ransac_outlier"
    HIGH_REPROJ_ERROR = "high_reproj_error"
    NOT_SAMPLED = "not_sampled"


@dataclass
class UncertaintyMeasures:
    """Uncertainty signals at each stage"""
    
    # Stage 1: Extraction
    detector_score: float = 0.0  # XFeat reliability
    detector_rank: int = 0  # Rank among extracted features
    
    # Stage 2: Quantization
    quantization_error: float = 0.0  # L2 distance before/after quantization
    descriptor_norm_fp32: float = 0.0
    descriptor_norm_uint8: float = 0.0
    
    # Stage 3: Vocabulary
    vocab_word_id: int = -1
    distance_to_word_center: float = 0.0
    distance_to_2nd_nearest_word: float = 0.0  # Ambiguity measure
    word_boundary_margin: float = 0.0  # How close to decision boundary
    
    # Stage 4: Matching
    n_candidates: int = 0  # How many map points in same vocab word
    best_match_distance: float = np.inf
    second_best_match_distance: float = np.inf
    ratio_test_value: float = 0.0  # best / second_best
    
    # Stage 5: RANSAC
    was_sampled: bool = False
    inlier_count_when_sampled: int = 0
    reprojection_error: float = np.inf
    
    # Stage 6: Final pose
    contributed_to_pose: bool = False
    final_reprojection_error: float = np.inf


@dataclass
class FeatureLifecycle:
    """Complete lifecycle record of one feature"""
    
    # Identity
    frame_id: int
    frame_name: str
    feature_id: int  # Index in frame's feature list
    
    # Birth
    keypoint_2d: np.ndarray = field(default_factory=lambda: np.zeros(2))
    descriptor_fp32: np.ndarray = field(default_factory=lambda: np.zeros(64))
    descriptor_uint8: np.ndarray = field(default_factory=lambda: np.zeros(64, dtype=np.uint8))
    
    # Lifecycle
    current_stage: LifecycleStage = LifecycleStage.EXTRACTED
    death_cause: DeathCause = DeathCause.ALIVE
    
    # Uncertainty measures
    uncertainty: UncertaintyMeasures = field(default_factory=UncertaintyMeasures)
    
    # Outcome
    map_point_id: Optional[int] = None
    localization_successful: bool = False


class FeatureLifecycleLogger:
    """Logs feature lifecycles during localization"""
    
    def __init__(self):
        self.features: List[FeatureLifecycle] = []
        self.current_frame_id = 0
        self.current_frame_name = ""
        
    def start_frame(self, frame_id: int, frame_name: str):
        """Begin logging a new frame"""
        self.current_frame_id = frame_id
        self.current_frame_name = frame_name
        
    def log_extraction(self, keypoints: np.ndarray, descriptors_fp32: np.ndarray, 
                      scores: np.ndarray) -> List[FeatureLifecycle]:
        """
        Log Stage 1: Feature extraction
        
        Returns: List of FeatureLifecycle objects for this frame
        """
        frame_features = []
        
        # Sort by detector score to get ranks
        score_ranks = np.argsort(-scores)  # Descending
        
        for i in range(len(keypoints)):
            feature = FeatureLifecycle(
                frame_id=self.current_frame_id,
                frame_name=self.current_frame_name,
                feature_id=i,
                keypoint_2d=keypoints[i].copy(),
                descriptor_fp32=descriptors_fp32[i].copy()
            )
            
            # Uncertainty: Detector confidence
            feature.uncertainty.detector_score = float(scores[i])
            feature.uncertainty.detector_rank = int(np.where(score_ranks == i)[0][0])
            feature.uncertainty.descriptor_norm_fp32 = float(np.linalg.norm(descriptors_fp32[i]))
            
            frame_features.append(feature)
            
        self.features.extend(frame_features)
        return frame_features
    
    def log_quantization(self, frame_features: List[FeatureLifecycle], 
                        descriptors_uint8: np.ndarray):
        """
        Log Stage 2: Quantization (FP32 → UINT8)
        
        Uncertainty: Information loss during quantization
        """
        for i, feature in enumerate(frame_features):
            feature.descriptor_uint8 = descriptors_uint8[i].copy()
            feature.current_stage = LifecycleStage.QUANTIZED
            
            # Measure quantization error
            # Re-normalize uint8 back to fp32 range to compare
            desc_reconstructed = (descriptors_uint8[i].astype(np.float32) / 255.0) - 0.5
            quantization_error = np.linalg.norm(feature.descriptor_fp32 - desc_reconstructed)
            
            feature.uncertainty.quantization_error = float(quantization_error)
            feature.uncertainty.descriptor_norm_uint8 = float(np.linalg.norm(descriptors_uint8[i]))
    
    def log_vocab_assignment(self, frame_features: List[FeatureLifecycle],
                            word_ids: np.ndarray,
                            distances_to_center: np.ndarray,
                            distances_to_2nd_word: Optional[np.ndarray] = None):
        """
        Log Stage 3: Vocabulary tree assignment
        
        Uncertainty: Ambiguity (how close to boundary between words)
        """
        for i, feature in enumerate(frame_features):
            feature.uncertainty.vocab_word_id = int(word_ids[i])
            feature.uncertainty.distance_to_word_center = float(distances_to_center[i])
            
            if distances_to_2nd_word is not None:
                feature.uncertainty.distance_to_2nd_nearest_word = float(distances_to_2nd_word[i])
                # Boundary margin: how much closer to 1st vs 2nd word
                margin = distances_to_2nd_word[i] - distances_to_center[i]
                feature.uncertainty.word_boundary_margin = float(margin)
            
            feature.current_stage = LifecycleStage.VOCAB_ASSIGNED
    
    def log_matching(self, frame_features: List[FeatureLifecycle],
                    query_indices: np.ndarray,
                    map_indices: np.ndarray,
                    distances: np.ndarray,
                    n_candidates_per_query: Dict[int, int]):
        """
        Log Stage 4: Descriptor matching
        
        Uncertainty: Ratio test margin, number of ambiguous candidates
        """
        # Build mapping: query_idx → [(map_idx, distance), ...]
        query_matches = {}
        for q_idx, m_idx, dist in zip(query_indices, map_indices, distances):
            if q_idx not in query_matches:
                query_matches[q_idx] = []
            query_matches[q_idx].append((m_idx, float(dist)))
        
        # Sort by distance for ratio test
        for q_idx in query_matches:
            query_matches[q_idx] = sorted(query_matches[q_idx], key=lambda x: x[1])
        
        # Log for each feature
        for i, feature in enumerate(frame_features):
            n_cand = n_candidates_per_query.get(i, 0)
            feature.uncertainty.n_candidates = n_cand
            
            if i in query_matches and len(query_matches[i]) > 0:
                matches = query_matches[i]
                best_m_idx, best_dist = matches[0]
                
                feature.uncertainty.best_match_distance = best_dist
                feature.map_point_id = int(best_m_idx)
                feature.current_stage = LifecycleStage.MATCHED
                
                if len(matches) > 1:
                    second_m_idx, second_dist = matches[1]
                    feature.uncertainty.second_best_match_distance = second_dist
                    # Ratio test: best / second_best (lower is better)
                    feature.uncertainty.ratio_test_value = best_dist / (second_dist + 1e-8)
                    feature.current_stage = LifecycleStage.RATIO_TESTED
                else:
                    # Only one match - perfect ratio test
                    feature.uncertainty.second_best_match_distance = best_dist * 2.0
                    feature.uncertainty.ratio_test_value = 0.5  # Arbitrary good value
                    feature.current_stage = LifecycleStage.RATIO_TESTED
            else:
                # No matches found
                feature.death_cause = DeathCause.NO_CANDIDATE
    
    def log_ransac(self, frame_features: List[FeatureLifecycle],
                  inlier_indices: np.ndarray,
                  reprojection_errors: np.ndarray,
                  all_tested_indices: Optional[np.ndarray] = None):
        """
        Log Stage 5: RANSAC geometric verification
        
        Uncertainty: Reprojection error, inlier/outlier status
        """
        inlier_set = set(inlier_indices.flatten())
        
        if all_tested_indices is not None:
            tested_set = set(all_tested_indices)
        else:
            tested_set = set(range(len(frame_features)))
        
        # Build mapping from feature index to reprojection error
        reproj_error_map = {}
        if all_tested_indices is not None and len(reprojection_errors) == len(all_tested_indices):
            for idx, error in zip(all_tested_indices, reprojection_errors):
                reproj_error_map[idx] = error
        
        for i, feature in enumerate(frame_features):
            if feature.map_point_id is None:
                continue  # Didn't get matched
            
            if i in tested_set:
                feature.uncertainty.was_sampled = True
                feature.current_stage = LifecycleStage.RANSAC_SAMPLED
                
                # Assign reprojection error if available
                if i in reproj_error_map:
                    feature.uncertainty.reprojection_error = float(reproj_error_map[i])
                
                if i in inlier_set:
                    feature.current_stage = LifecycleStage.RANSAC_INLIER
                else:
                    feature.death_cause = DeathCause.RANSAC_OUTLIER
            else:
                feature.uncertainty.was_sampled = False
                feature.death_cause = DeathCause.NOT_SAMPLED
    
    def log_final_pose(self, frame_features: List[FeatureLifecycle],
                      final_inliers: np.ndarray,
                      final_reproj_errors: np.ndarray,
                      localization_success: bool):
        """
        Log Stage 6: Final pose contribution
        
        Uncertainty: Final reprojection error
        """
        final_inlier_set = set(final_inliers.flatten())
        
        # Build mapping from feature index to final reprojection error
        final_reproj_map = {}
        if len(final_inliers) > 0 and len(final_reproj_errors) == len(final_inliers):
            for idx, error in zip(final_inliers.flatten(), final_reproj_errors):
                final_reproj_map[idx] = error
        
        for i, feature in enumerate(frame_features):
            if i in final_inlier_set:
                feature.current_stage = LifecycleStage.POSE_CONTRIBUTOR
                feature.uncertainty.contributed_to_pose = True
                if i in final_reproj_map:
                    feature.uncertainty.final_reprojection_error = float(final_reproj_map[i])
            
            feature.localization_successful = localization_success
    
    def get_survivors(self) -> List[FeatureLifecycle]:
        """Return features that successfully contributed to localization"""
        return [f for f in self.features 
                if f.current_stage == LifecycleStage.POSE_CONTRIBUTOR 
                and f.localization_successful]
    
    def get_dead_at_stage(self, stage: LifecycleStage) -> List[FeatureLifecycle]:
        """Return features that died at a specific stage"""
        return [f for f in self.features 
                if f.current_stage == stage 
                and f.death_cause != DeathCause.ALIVE]
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """Export all logged features to pandas DataFrame for analysis"""
        records = []
        
        for feature in self.features:
            record = {
                'frame_id': feature.frame_id,
                'frame_name': feature.frame_name,
                'feature_id': feature.feature_id,
                'keypoint_x': feature.keypoint_2d[0],
                'keypoint_y': feature.keypoint_2d[1],
                
                # Lifecycle
                'final_stage': feature.current_stage.value,
                'death_cause': feature.death_cause.value,
                'survived': feature.current_stage == LifecycleStage.POSE_CONTRIBUTOR,
                'localization_successful': feature.localization_successful,
                
                # Uncertainty measures
                'detector_score': feature.uncertainty.detector_score,
                'detector_rank': feature.uncertainty.detector_rank,
                'quantization_error': feature.uncertainty.quantization_error,
                'vocab_word_id': feature.uncertainty.vocab_word_id,
                'distance_to_word_center': feature.uncertainty.distance_to_word_center,
                'word_boundary_margin': feature.uncertainty.word_boundary_margin,
                'n_candidates': feature.uncertainty.n_candidates,
                'best_match_distance': feature.uncertainty.best_match_distance,
                'ratio_test_value': feature.uncertainty.ratio_test_value,
                'was_sampled_ransac': feature.uncertainty.was_sampled,
                'reprojection_error': feature.uncertainty.reprojection_error,
                'final_reprojection_error': feature.uncertainty.final_reprojection_error,
            }
            
            records.append(record)
        
        return pd.DataFrame(records)
    
    def save(self, filepath: str):
        """Save logged data to CSV"""
        df = self.export_to_dataframe()
        df.to_csv(filepath, index=False)
        print(f"Saved {len(df)} feature lifecycles to {filepath}")
    
    def print_summary(self):
        """Print summary statistics"""
        total = len(self.features)
        if total == 0:
            print("No features logged")
            return
        
        survivors = self.get_survivors()
        
        print(f"\n{'='*60}")
        print(f"FEATURE LIFECYCLE SUMMARY")
        print(f"{'='*60}")
        print(f"Total features extracted: {total}")
        print(f"Survivors (contributed to pose): {len(survivors)} ({len(survivors)/total*100:.1f}%)")
        print(f"")
        
        # Death causes
        print(f"Death causes:")
        for cause in DeathCause:
            if cause == DeathCause.ALIVE:
                continue
            count = sum(1 for f in self.features if f.death_cause == cause)
            if count > 0:
                print(f"  {cause.value:30s}: {count:5d} ({count/total*100:.1f}%)")
        
        # Stage reached
        print(f"\nFarthest stage reached:")
        for stage in LifecycleStage:
            count = sum(1 for f in self.features if f.current_stage == stage)
            if count > 0:
                print(f"  {stage.value:30s}: {count:5d} ({count/total*100:.1f}%)")