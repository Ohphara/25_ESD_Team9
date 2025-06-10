#include <opencv2/opencv.hpp>
#include "yolo11n.h"
#include "./bytetrack/include/BYTETracker.h"
#include <unordered_map>
#include <set>
#include <vector>
#include <string>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <fstream>

using namespace std;

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
const int FRAME_SIZE = 480;
const float TOP_DIST_M = 4.0;
const float BOTTOM_DIST_M = 2.5;
const float DEFAULT_SPEED_THRESH = 1.5;

const std::set<int> ALLOWED_CLASSES = {0, 1, 2, 3, 5, 6, 7};
const std::unordered_map<int, float> SPEED_THRESHOLDS = {
    {0, 3.0f},  // person
    {1, 5.0f},  // bicycle
    {2, 7.0f},  // car
    {3, 5.0f},  // motorcycle
    {5, 7.0f},  // bus
    {6, 7.0f},  // train
    {7, 7.0f}   // truck
};

// prev info struct
struct PrevInfo {
    int cx, cy;
    double t_prev;
    float speed;
    bool prev_warn_candidate;
};
std::unordered_map<int, PrevInfo> prev_info;
std::unordered_map<int, double> alert_cooldowns;

// 거리 추정 (y 좌표 기반 선형보간)
float estimate_distance_from_y(int y, int top_y, int bottom_y) {
    return ((float)(y - top_y) / (float)(bottom_y - top_y)) * (BOTTOM_DIST_M - TOP_DIST_M) + TOP_DIST_M;
}

// 위험도 점수 계산 (심플 버전)
float compute_risk_score(float speed, float distance, int class_id, float approach_angle) {
    float dist_score = (approach_angle < 0.5f) ? std::max(0.0f, 5.0f - distance)
                                               : ((distance < 2.5f) ? 2.0f : 0.0f);
    float speed_score = std::min(speed / 2.0f, 5.0f);
    float obj_score = 1.0f;
    // 클래스별 가중치 python과 맞춤
    switch(class_id) {
        case 2: case 5: case 6: case 7: obj_score = 3.0f; break; // car, bus, train, truck
        case 1: obj_score = 1.0f; break; // bicycle
        case 3: obj_score = 2.0f; break; // motorcycle
        case 0: obj_score = 0.0f; break; // person
        default: obj_score = 1.0f; break;
    }
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
    std::string input_source = "cam:8";
    bool save_log = false;
    std::string log_path = "alert_log.csv";

    if (argc > 1) input_source = argv[1];
    if (argc > 2) {
        std::string opt = argv[2];
        if (opt == "--save_log" || opt == "1" || opt == "True") save_log = true;
    }
    if (argc > 3) log_path = argv[3];

    cv::VideoCapture cap;
    if (input_source.rfind("cam:", 0) == 0) {
        int cam_idx = std::stoi(input_source.substr(4));
        cap.open(cam_idx);
    } else {
        cap.open(input_source);
    }
    if (!cap.isOpened()) {
        fprintf(stderr, "No camera or video file\n");
        return -1;
    }

    std::ofstream log_file;
    if (save_log) {
        log_file.open(log_path);
        log_file << "frame,id,class,speed,distance_m,risk_score,cx,cy,y2,in_zone" << std::endl;
    }

    int frame_id = 0, frame_rate = 30;
    BYTETracker tracker(frame_rate, 30);
    cv::Mat frame;
    double total_elapsed_ms = 0.0;
    int frame_count = 0;

    // 사다리꼴 zone 좌표
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
    // 사다리꼴 중심
    cv::Point zone_center(
        int((zone_pts[0].x + zone_pts[1].x + zone_pts[2].x + zone_pts[3].x) / 4.0),
        int((zone_pts[0].y + zone_pts[1].y + zone_pts[2].y + zone_pts[3].y) / 4.0)
    );

    while (cap.read(frame)) {
        int64 t0 = cv::getTickCount();
        cv::resize(frame, frame, cv::Size(FRAME_SIZE, FRAME_SIZE));

        std::vector<Object> objects;
        detect_yolo11(frame, objects);

        std::vector<STrack> tracks = tracker.update(objects);

        // draw zone
        cv::polylines(frame, zone_pts, true, cv::Scalar(0,255,0), 1);

        double t_now = (double)cv::getTickCount() / cv::getTickFrequency();

        for (const auto& track : tracks) {
            const std::vector<float>& tlwh = track.tlwh;
            int track_id = track.track_id;
            int x1 = int(tlwh[0]), y1 = int(tlwh[1]);
            int x2 = x1 + int(tlwh[2]), y2 = y1 + int(tlwh[3]);
            int cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;

            // bbox와 detection 매칭 (가까운 center)
            int class_id = -1;
            float prob = 0.0f;
            for (const auto& obj : objects) {
                if (abs(cx - (obj.rect.x + obj.rect.width/2)) < 10 && abs(cy - (obj.rect.y + obj.rect.height/2)) < 10) {
                    class_id = obj.label;
                    prob = obj.prob;
                    break;
                }
            }
            if (class_id == -1) continue;
            if (ALLOWED_CLASSES.count(class_id) == 0) continue;

            float speed = 0, approach_angle = 0;
            bool prev_warn_candidate = false;
            if (prev_info.count(track_id)) {
                auto& p = prev_info[track_id];
                float dt = t_now - p.t_prev;
                float dx = cx - p.cx, dy = cy - p.cy;
                float dist = sqrt(dx*dx + dy*dy);
                speed = (dt > 0.01) ? dist / dt : 0;
                if (fabs(speed - p.speed) > 30) speed = p.speed;
                approach_angle = fabs(dx) / (fabs(dy) + 1e-6);
                prev_warn_candidate = p.prev_warn_candidate;
            }
            // 사다리꼴 zone 안/밖
            bool in_zone = cv::pointPolygonTest(zone_pts, cv::Point(cx, y2), false) >= 0;
            float speed_thresh = SPEED_THRESHOLDS.count(class_id) ? SPEED_THRESHOLDS.at(class_id) : DEFAULT_SPEED_THRESH;
            bool current_warn_candidate = in_zone && speed > speed_thresh;

            bool warn = false;
            float risk_score = 0.0f;
            if (current_warn_candidate && prev_warn_candidate) {
                float distance_m = estimate_distance_from_y(y2, top_y, bottom_y);
                risk_score = compute_risk_score(speed, distance_m, class_id, approach_angle);
                if (should_alert(track_id, risk_score, t_now)) {
                    warn = true;
                    std::string class_name = (class_id >= 0 && class_id < 80) ? class_names[class_id] : "object";
                    printf("ALERT: %s is approaching.\n", class_name.c_str());
                    fflush(stdout);
                    if (save_log && log_file.is_open()) {
                        log_file << frame_id << "," << track_id << "," << class_name << ","
                                 << speed << "," << distance_m << "," << risk_score << ","
                                 << cx << "," << cy << "," << y2 << "," << int(in_zone) << std::endl;
                    }
                }
            }
            prev_info[track_id] = {cx, cy, t_now, speed, current_warn_candidate};

            // 시각화: bbox/label/노란선
            cv::rectangle(frame, cv::Rect(x1, y1, x2-x1, y2-y1), warn ? cv::Scalar(0,0,255) : cv::Scalar(255,255,0), 2);
            char label[128];
            sprintf(label, "%s %.1fpx/s%s", (class_id >= 0 && class_id < 80) ? class_names[class_id] : "object", speed, warn ? " ⚠" : "");
            cv::putText(frame, label, cv::Point(x1, y1-10), cv::FONT_HERSHEY_SIMPLEX, 0.5, warn ? cv::Scalar(0,0,255) : cv::Scalar(255,255,0), 2);
            // bbox 하단 중앙에서 zone 중심으로 노란선
            cv::line(frame, cv::Point(cx, y2), zone_center, cv::Scalar(0, 255, 255), 2);
        }

        // 프레임 표시
        cv::imshow("yolo11n", frame);

        int64 t1 = cv::getTickCount();
        double elapsed_ms = (t1 - t0) * 1000.0 / cv::getTickFrequency();
        total_elapsed_ms += elapsed_ms;
        frame_count++;

        if (!cap.isOpened() || frame.empty()) break;
        if (cv::waitKey(1) == 27) break; // ESC
        frame_id++;
    }

    if (frame_count > 0) {
        double avg_fps = 1000.0 / (total_elapsed_ms / frame_count);
        printf("Average FPS: %.2f\n", avg_fps);
    }
    if (log_file.is_open()) log_file.close();

    return 0;
}