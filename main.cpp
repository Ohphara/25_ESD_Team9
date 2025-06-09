#include <opencv2/opencv.hpp>
#include "yolo11n.h"
#include "./bytetrack/include/BYTETracker.h"
#include <unordered_map>
#include <vector>
#include <string>
#include <cstdio>
#include <cmath>
#include <iostream>

using namespace std;
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
// ----------- 경보(Warning) 파라미터 세팅 -----------
const std::unordered_map<int, float> SPEED_THRESHOLDS = {
    {1, 5.0},  // bicycle
    {21, 5.0}, // elephant (임의)
    {24, 2.0}, // backpack (임의)
    {22, 4.0}, // bear (임의)
    {14, 3.0}, // bird (임의)
};
const float DEFAULT_SPEED_THRESH = 1.5;
const int FRAME_SIZE = 480;
const float TOP_DIST_M = 4.0;
const float BOTTOM_DIST_M = 2.5;

struct PrevInfo {
    int cx, cy;
    double t_prev;
    float speed;
};

std::unordered_map<int, PrevInfo> prev_info;      // obj_id -> PrevInfo
std::unordered_map<int, double> alert_cooldowns;  // obj_id -> last_alert_time

// 거리 추정 (y 좌표 기반 선형보간)
float estimate_distance_from_y(int y, int top_y, int bottom_y) {
    return ((float)(y - top_y) / (float)(bottom_y - top_y)) * (BOTTOM_DIST_M - TOP_DIST_M) + TOP_DIST_M;
}

// 위험도 점수 계산 (심플 버전)
float compute_risk_score(float speed, float distance_m, int class_id, float approach_angle) {
    float dist_score = (approach_angle < 0.5) ? std::max(0.0f, 5.0f - distance_m) : ((distance_m < 2.5f) ? 2.0f : 0.0f);
    float speed_score = std::min(speed / 2.0f, 5.0f);
    float obj_score = 1.0f;
    if (class_id == 2 || class_id == 5 || class_id == 7) obj_score = 3.0f;  // car, bus, truck
    if (class_id == 1 || class_id == 3) obj_score = 2.0f; // bicycle, motorcycle
    return dist_score + speed_score + obj_score;
}

// 경보를 낼지 판단
bool should_alert(int obj_id, float risk_score, double t_now) {
    if (risk_score >= 10) return true;
    double cooldown = (risk_score >= 6) ? 2.0 : 5.0;
    double last_alert = alert_cooldowns[obj_id];
    if (t_now - last_alert > cooldown) {
        alert_cooldowns[obj_id] = t_now;
        return true;
    }
    return false;
}

void draw_objects(cv::Mat& image, const std::vector<Object>& objects,
                 const std::vector<cv::Point>& zone_pts, const cv::Point& zone_center)
{
    static cv::Scalar colors[] = {
        cv::Scalar(244, 67, 54), cv::Scalar(233, 30, 99), cv::Scalar(156, 39, 176), cv::Scalar(103, 58, 183),
        cv::Scalar(63, 81, 181), cv::Scalar(33, 150, 243), cv::Scalar(3, 169, 244), cv::Scalar(0, 188, 212),
        cv::Scalar(0, 150, 136), cv::Scalar(76, 175, 80), cv::Scalar(139, 195, 74), cv::Scalar(205, 220, 57),
        cv::Scalar(255, 235, 59), cv::Scalar(255, 193, 7), cv::Scalar(255, 152, 0), cv::Scalar(255, 87, 34),
        cv::Scalar(121, 85, 72), cv::Scalar(158, 158, 158), cv::Scalar(96, 125, 139)
    };
    // 사다리꼴 zone 먼저 그리기
    cv::polylines(image, zone_pts, true, cv::Scalar(0,255,0), 1);

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

        // bbox 하단 중앙 -> zone_center로 노란선
        int x_center = int(obj.rect.x + obj.rect.width/2);
        int y_bottom = int(obj.rect.y + obj.rect.height);
        cv::line(image, cv::Point(x_center, y_bottom), zone_center, cv::Scalar(0,255,255), 1);
    }
}

int main(int argc, char** argv) {
    cv::VideoCapture cap;
    // 기본값: 웹캠 8번
    if (argc > 1) {
        std::string arg1 = argv[1];
        if (arg1.rfind("cam:", 0) == 0) {
            int cam_idx = std::stoi(arg1.substr(4));
            cap.open(cam_idx);
        } else {
            // 동영상 파일 경로
            cap.open(arg1);
        }
    } else {
        cap.open(8); // default cam
    }

    if (!cap.isOpened()) {
        fprintf(stderr, "No camera or video file\n");
        return -1;
    }

    int frame_id = 0;
    int frame_rate = 30;
    BYTETracker tracker(frame_rate, 30);

    cv::Mat frame;
    double total_elapsed_ms = 0.0;
    int frame_count = 0;

    // 사다리꼴 구역 설정
    int top_y = int(FRAME_SIZE * 0.55);
    int bottom_y = int(FRAME_SIZE * 0.95);
    int center_x = FRAME_SIZE / 2;
    int top_width = 100, bottom_width = 220;
    std::vector<cv::Point> zone_pts = {
        cv::Point(center_x - top_width/2, top_y),
        cv::Point(center_x + top_width/2, top_y),
        cv::Point(center_x + bottom_width/2, bottom_y),
        cv::Point(center_x - bottom_width/2, bottom_y)
    };
    cv::Point zone_center(center_x, bottom_y);

    while (cap.read(frame)) {
        int64 t0 = cv::getTickCount();

        cv::resize(frame, frame, cv::Size(FRAME_SIZE, FRAME_SIZE)); // 강제 resize

        std::vector<Object> objects;
        detect_yolo11(frame, objects);

        // 트래킹
        std::vector<STrack> tracks = tracker.update(objects);

        // === draw_objects에서 모든 시각화: bbox, 라벨, 사다리꼴, 선 ===
        draw_objects(frame, objects, zone_pts, zone_center);

        // 경보 로직
        double t_now = (double)cv::getTickCount() / cv::getTickFrequency();
        for (const auto& track : tracks) {
            const std::vector<float>& tlwh = track.tlwh;
            int track_id = track.track_id;

            // bbox center 계산
            int x1 = int(tlwh[0]), y1 = int(tlwh[1]);
            int x2 = x1 + int(tlwh[2]), y2 = y1 + int(tlwh[3]);
            int cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
            int class_id = -1;
            float prob = 0.0f;
            // detection 매칭
            for (const auto& obj : objects) {
                if (abs(cx - (obj.rect.x + obj.rect.width/2)) < 10 && abs(cy - (obj.rect.y + obj.rect.height/2)) < 10) {
                    class_id = obj.label;
                    prob = obj.prob;
                    break;
                }
            }
            if (class_id == -1) continue;

            float speed = 0, approach_angle = 0;
            if (prev_info.count(track_id)) {
                auto& p = prev_info[track_id];
                float dt = t_now - p.t_prev;
                float dx = cx - p.cx, dy = cy - p.cy;
                float dist = sqrt(dx*dx + dy*dy);
                speed = dt > 0.01 ? dist / dt : 0;
                if (fabs(speed - p.speed) > 30) speed = p.speed;
                approach_angle = fabs(dx) / (fabs(dy) + 1e-6);
            }
            prev_info[track_id] = {cx, cy, t_now, speed};

            // 아래 중앙점 기준 거리 추정
            float distance_m = estimate_distance_from_y(y2, top_y, bottom_y);

            // 사다리꼴 구역 안/밖 판정
            int in_zone = cv::pointPolygonTest(zone_pts, cv::Point(cx, y2), false) >= 0;
            float speed_thresh = SPEED_THRESHOLDS.count(class_id) ? SPEED_THRESHOLDS.at(class_id) : DEFAULT_SPEED_THRESH;

            if (in_zone && speed > speed_thresh) {
                float risk_score = compute_risk_score(speed, distance_m, class_id, approach_angle);
                if (should_alert(track_id, risk_score, t_now)) {
                    std::string class_name = (class_id >= 0 && class_id < 80) ? class_names[class_id] : "object";
                    printf("ALERT: %s is approaching.\n", class_name.c_str());
                    fflush(stdout);
                }
            }
        }

        // === 프레임 표시 ===
        cv::imshow("yolo11n", frame);

        int64 t1 = cv::getTickCount();
        double elapsed_ms = (t1 - t0) * 1000.0 / cv::getTickFrequency();
        total_elapsed_ms += elapsed_ms;
        frame_count++;

        if (!cap.isOpened() || frame.empty())
            break;

        if (cv::waitKey(1) == 27) break; // ESC
        frame_id++;
    }

    if (frame_count > 0) {
        double avg_fps = 1000.0 / (total_elapsed_ms / frame_count);
        printf("Average FPS: %.2f\n", avg_fps);
    }

    return 0;
}
