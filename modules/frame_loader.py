import os
import cv2


class FrameLoader:
    def __init__(self, images_path, max_images=None):
        self.images_path = images_path
        self.max_images = max_images

    def get_frames(self):
        """Load images from already-rectified EuRoC dataset"""
        images = sorted([os.path.join(self.images_path[0], f)
                              for f in os.listdir(self.images_path[0])
                              if f.endswith(('.png', '.jpg', '.jpeg'))])
    

        if self.max_images is not None:
            images = images[:self.max_images]

        for image_path in images:
            image_frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            

            if image_frame is None:
                print(f"Warning: Could not load images {image_path}")
                continue

            yield image_frame

    # def get_frame_pairs_with_timestamps(self):
    #     """Load stereo pairs with timestamp information from EuRoC"""
    #     left_images = sorted([f for f in os.listdir(self.images_path[0])
    #                           if f.endswith(('.png', '.jpg', '.jpeg'))])
    #     right_images = sorted([f for f in os.listdir(self.images_path[1])
    #                            if f.endswith(('.png', '.jpg', '.jpeg'))])

    #     if self.max_images is not None:
    #         left_images = left_images[:self.max_images]
    #         right_images = right_images[:self.max_images]

    #     for left_filename, right_filename in zip(left_images, right_images):
    #         left_path = os.path.join(self.images_path[0], left_filename)
    #         right_path = os.path.join(self.images_path[1], right_filename)

    #         left_frame = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    #         right_frame = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

    #         if left_frame is None or right_frame is None:
    #             continue

    #         timestamp = left_filename.split('.')[0]  # Assumes format: <timestamp>.png

    #         yield left_frame, right_frame, timestamp


    # def get_frame_pairs_for_unrectified(self):
    #     left_images = sorted([os.path.join(self.images_path[0], f)
    #                           for f in os.listdir(self.images_path[0])
    #                           if f.endswith(('.png', '.jpg', '.jpeg'))])
    #     right_images = sorted([os.path.join(self.images_path[1], f)
    #                            for f in os.listdir(self.images_path[1])
    #                            if f.endswith(('.png', '.jpg', '.jpeg'))])

    #     if self.max_images is not None:
    #         left_images = left_images[:self.max_images]
    #         right_images = right_images[:self.max_images]

    #     for left_path, right_path in zip(left_images, right_images):
    #         left_frame = cv2.imread(left_path, cv2.IMREAD_COLOR)
    #         right_frame = cv2.imread(right_path, cv2.IMREAD_COLOR)

    #         if left_frame is None or right_frame is None:
    #             continue

    #         left_rect, right_rect = self.rectifier.rectify_stereo_pair(left_frame, right_frame)
    #         yield left_rect, right_rect