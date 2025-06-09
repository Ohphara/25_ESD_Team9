#include <opencv2/opencv.hpp>
#include "yolo11n.h"
#include "./bytetrack/include/BYTETracker.h"

void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects) {
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };
    static cv::Scalar colors[] = {
        cv::Scalar(244, 67, 54),
        cv::Scalar(233, 30, 99),
        cv::Scalar(156, 39, 176),
        cv::Scalar(103, 58, 183),
        cv::Scalar(63, 81, 181),
        cv::Scalar(33, 150, 243),
        cv::Scalar(3, 169, 244),
        cv::Scalar(0, 188, 212),
        cv::Scalar(0, 150, 136),
        cv::Scalar(76, 175, 80),
        cv::Scalar(139, 195, 74),
        cv::Scalar(205, 220, 57),
        cv::Scalar(255, 235, 59),
        cv::Scalar(255, 193, 7),
        cv::Scalar(255, 152, 0),
        cv::Scalar(255, 87, 34),
        cv::Scalar(121, 85, 72),
        cv::Scalar(158, 158, 158),
        cv::Scalar(96, 125, 139)
    };
    cv::Mat image = bgr.clone();
    for (size_t i = 0; i < objects.size(); i++) {
        const Object& obj = objects[i];
        const cv::Scalar& color = colors[i % 19];
        cv::rectangle(image, obj.rect, color, 2);
        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0) y = 0;
        if (x + label_size.width > image.cols) x = image.cols - label_size.width;
        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cv::Scalar(255,255,255), -1);
        cv::putText(image, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0));
    }
    cv::imshow("yolo11n", image);
}

int main() {
    int cam_index = 8; // 카메라 번호
    cv::VideoCapture cap(cam_index);
    if (!cap.isOpened()) {
        fprintf(stderr, "No camera\n");
        return -1;
    }
    int frame_id = 0;
    int frame_rate = 30;
    BYTETracker tracker(frame_rate, 30);

    cv::Mat frame;
    double total_elapsed_ms = 0.0; // 전체 처리 시간 누적

    while (cap.read(frame)) {
        int64 t0 = cv::getTickCount();

        std::vector<Object> objects;
        detect_yolo11(frame, objects);
		for (const auto& obj : objects) {
			printf("[DET] label:%d prob:%.2f x:%.1f y:%.1f w:%.1f h:%.1f\n",
				obj.label, obj.prob, obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
		}
        // ---- Tracking 결과 로그 및 화면 표시 ----
        std::vector<STrack> output_stracks = tracker.update(objects);
        printf("=== Tracking Results [Frame %d] ===\n", frame_id);
        for (const auto& track : output_stracks) {
            const std::vector<float>& tlwh = track.tlwh;
            int track_id = track.track_id;

            // (클래스 ID 매칭은 상황에 맞게 추가)
            int class_id = -1;
            if (!objects.empty()) {
                class_id = objects[0].label; // 예시: 첫 detection의 클래스
            }
            printf("ID:%d | BBOX:[%.1f, %.1f, %.1f, %.1f] | Class:%d\n",
                track_id, tlwh[0], tlwh[1], tlwh[2], tlwh[3], class_id);

            // 화면 표시
            cv::rectangle(frame, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), cv::Scalar(0,255,0), 2);
            char text[64];
            snprintf(text, sizeof(text), "id:%d", track_id);
            cv::putText(frame, text, cv::Point(tlwh[0], tlwh[1]-5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,255), 2);
        }

        // ---- Detection 결과 로그 ----
        printf("--- Detection Results [Frame %d] ---\n", frame_id);
        for (const auto& obj : objects) {
            printf("Class:%d | Prob:%.2f | BBOX:[%.1f, %.1f, %.1f, %.1f]\n",
                obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.x + obj.rect.width, obj.rect.y + obj.rect.height);
        }

        int64 t1 = cv::getTickCount();
        double elapsed_ms = (t1 - t0) * 1000.0 / cv::getTickFrequency();
        total_elapsed_ms += elapsed_ms;
        double fps = 1000.0 / elapsed_ms;

        printf("[frame %d] time = %.2f ms, FPS = %.2f, Tracks = %zu\n\n", 
            frame_id, elapsed_ms, fps, output_stracks.size());

        frame_id++;
        fflush(stdout);

        cv::imshow("tracking", frame);
        if (cv::waitKey(1) == 27) break; // ESC
    }

    if (frame_id > 0) {
        printf("=== Total average FPS: %.2f ===\n", frame_id * 1000.0 / total_elapsed_ms);
    }
    return 0;
}
