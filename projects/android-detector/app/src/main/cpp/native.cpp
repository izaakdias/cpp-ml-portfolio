#include <jni.h>

#include <algorithm>
#include <cmath>
#include <vector>
namespace {
void invalid(JNIEnv* env, const char* message) {
  env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), message);
}
struct Plane {
  const unsigned char* data;
  jlong length;
  int row, pixel;
};
Plane plane(JNIEnv* env, jobject buffer, int row, int pixel) {
  return {static_cast<const unsigned char*>(env->GetDirectBufferAddress(buffer)),
          env->GetDirectBufferCapacity(buffer), row, pixel};
}
bool valid(const Plane& p, int width, int height) {
  return p.data && p.row > 0 && p.pixel > 0 &&
         p.length >
             static_cast<jlong>(height - 1) * p.row + static_cast<jlong>(width - 1) * p.pixel;
}
float overlap(const float* a, const float* b) {
  float h = std::max(0.f, std::min(a[4], b[4]) - std::max(a[2], b[2])),
        w = std::max(0.f, std::min(a[5], b[5]) - std::max(a[3], b[3]));
  float intersection = h * w, areaA = (a[4] - a[2]) * (a[5] - a[3]),
        areaB = (b[4] - b[2]) * (b[5] - b[3]);
  return intersection / std::max(1e-8f, areaA + areaB - intersection);
}
}  // namespace
extern "C" JNIEXPORT jbyteArray JNICALL Java_dev_izaak_portfolio_detector_NativePixels_rgb(
    JNIEnv* env, jobject, jobject y, jobject u, jobject v, jint width, jint height,
    jint left, jint top, jint right, jint bottom, jint yr,
    jint ur, jint vr, jint up, jint vp, jint rotation) {
  if (width < 2 || height < 2 || width > 8192 || height > 8192 ||
      (rotation != 0 && rotation != 90 && rotation != 180 && rotation != 270)) {
    invalid(env, "Invalid frame dimensions/rotation");
    return nullptr;
  }
  Plane yp = plane(env, y, yr, 1), uv = plane(env, u, ur, up), vv = plane(env, v, vr, vp);
  if(left<0||top<0||right>width||bottom>height||right<=left||bottom<=top){invalid(env,"Invalid crop rectangle");return nullptr;}
  if (!valid(yp, width, height) || !valid(uv, (width + 1) / 2, (height + 1) / 2) ||
      !valid(vv, (width + 1) / 2, (height + 1) / 2)) {
    invalid(env, "Invalid YUV plane strides/capacity");
    return nullptr;
  }
  int cw=right-left,ch=bottom-top;
  int rw = rotation % 180 ? ch : cw, rh = rotation % 180 ? cw : ch;
  std::vector<jbyte> output(300 * 300 * 3);
  for (int oy = 0; oy < 300; ++oy)
    for (int ox = 0; ox < 300; ++ox) {
      int rx = std::min(rw - 1, (2 * ox + 1) * rw / 600),
          ry = std::min(rh - 1, (2 * oy + 1) * rh / 600), sx = rx, sy = ry;
      if (rotation == 90) {
        sx = ry;
        sy = ch - 1 - rx;
      } else if (rotation == 180) {
        sx = cw - 1 - rx;
        sy = ch - 1 - ry;
      } else if (rotation == 270) {
        sx = cw - 1 - ry;
        sy = rx;
      }
      sx+=left;sy+=top;
      int yy = std::max(0, int(yp.data[sy * yp.row + sx]) - 16),
          uu = int(uv.data[(sy / 2) * uv.row + (sx / 2) * uv.pixel]) - 128,
          vvvalue = int(vv.data[(sy / 2) * vv.row + (sx / 2) * vv.pixel]) - 128;
      int channels[] = {(298 * yy + 409 * vvvalue + 128) >> 8,
                        (298 * yy - 100 * uu - 208 * vvvalue + 128) >> 8,
                        (298 * yy + 516 * uu + 128) >> 8};
      for (int c = 0; c < 3; ++c)
        output[(oy * 300 + ox) * 3 + c] = static_cast<jbyte>(std::clamp(channels[c], 0, 255));
    }
  auto result = env->NewByteArray(output.size());
  if (result)
    env->SetByteArrayRegion(result, 0, output.size(), output.data());
  return result;
}
extern "C" JNIEXPORT jfloatArray JNICALL Java_dev_izaak_portfolio_detector_NativePixels_filter(
    JNIEnv* env, jobject, jfloatArray boxes, jfloatArray classes, jfloatArray scores, jint count,
    jfloat threshold) {
  if (count < 0 || count > 10000 || env->GetArrayLength(boxes) < 4 * count ||
      env->GetArrayLength(classes) < count || env->GetArrayLength(scores) < count ||
      !std::isfinite(threshold) || threshold < 0 || threshold > 1) {
    invalid(env, "Invalid detection tensors");
    return nullptr;
  }
  std::vector<float> b(count * 4), c(count), s(count);
  env->GetFloatArrayRegion(boxes, 0, count * 4, b.data());
  env->GetFloatArrayRegion(classes, 0, count, c.data());
  env->GetFloatArrayRegion(scores, 0, count, s.data());
  std::vector<std::vector<float>> candidates;
  for (int i = 0; i < count; ++i) {
    bool finite = std::isfinite(c[i]) && std::isfinite(s[i]);
    for (int j = 0; j < 4; ++j) finite = finite && std::isfinite(b[4 * i + j]);
    if (!finite || s[i] < threshold || c[i] < 1)
      continue;
    std::vector<float> d = {c[i], s[i]};
    for (int j = 0; j < 4; ++j) d.push_back(std::clamp(b[4 * i + j], 0.f, 1.f));
    if (d[4] > d[2] && d[5] > d[3])
      candidates.push_back(std::move(d));
  }
  std::stable_sort(candidates.begin(), candidates.end(),
                   [](const auto& a, const auto& b) { return a[1] > b[1]; });
  std::vector<float> output;
  for (const auto& candidate : candidates) {
    bool suppressed = false;
    for (std::size_t i = 0; i < output.size(); i += 6)
      if (output[i] == candidate[0] && overlap(output.data() + i, candidate.data()) > .5f) {
        suppressed = true;
        break;
      }
    if (!suppressed)
      output.insert(output.end(), candidate.begin(), candidate.end());
    if (output.size() >= 30 * 6)
      break;
  }
  auto result = env->NewFloatArray(output.size());
  if (result)
    env->SetFloatArrayRegion(result, 0, output.size(), output.data());
  return result;
}
