#include "pr.hpp"
#include <iostream>
#include <memory>

float f(float x, float y)
{
    return x * x / 20.0f + y * y;
}
float dfdx( float x, float y )
{
    return x / 10.0f;
}
float dfdy(float x, float y)
{
    return 2.0f * y;
}

int main() {
    using namespace pr;

    Config config;
    config.ScreenWidth = 1920;
    config.ScreenHeight = 1080;
    config.SwapInterval = 1;
    Initialize(config);

    Camera3D camera;
    camera.origin = { 4, 4, 4 };
    camera.lookat = { 0, 0, 0 };
    camera.zUp = true;

    double e = GetElapsedTime();

    struct OptimizePoint
    {
        glm::vec2 p;
        std::vector<glm::vec2> ph;

        glm::vec2 adam_p;
        glm::vec2 adam_m;
        glm::vec2 adam_v;
        std::vector<glm::vec2> adam_ph;
    };

    int nSamples = 10;
    int iterations = 0;
    std::vector<OptimizePoint> points;

    auto initPoints = [&]() {
        points.clear();

        Xoshiro128StarStar random;
        for (int i = 0; i < nSamples; ++i)
        {
            OptimizePoint o = {};
            o.p.x = glm::mix(-2.0f, 2.0f, random.uniformf());
            o.p.y = glm::mix(-2.0f, 2.0f, random.uniformf());

            o.adam_p = o.p;
            o.adam_m = glm::vec2(0.0f, 0.0f);
            o.adam_v = glm::vec2(0.0f, 0.0f);

            points.push_back(o);
        }

        iterations = 0;
    };

    initPoints();

    while (pr::NextFrame() == false) {
        if (IsImGuiUsingMouse() == false) {
            UpdateCameraBlenderLike(&camera);
        }

        ClearBackground(0.1f, 0.1f, 0.1f, 1);

        BeginCamera(camera);

        PushGraphicState();

        DrawGrid(GridAxis::XY, 1.0f, 10, { 128, 128, 128 });
        DrawXYZAxis(1.0f);

        PrimBegin(PrimitiveMode::Points);
        int N = 50;
        LinearTransform i2p(0, N - 1, -3, 3);
        for( int ix = 0; ix < N; ix++)
        {
            for (int iy = 0; iy < N; iy++ )
            {
                float x = i2p(ix);
                float y = i2p(iy);
                float z = f(x, y);
                PrimVertex({ x, y, z }, { 255,255,255 });
            }
        }
        PrimEnd();

        for (int i = 0; i < points.size(); ++i)
        {
            OptimizePoint o = points[i];

            {
                float x = o.p.x;
                float y = o.p.y;
                float z = f(x, y);
                DrawSphere({ x, y,z }, 0.05, { 0, 255,255 });
            }
            {
                float x = o.adam_p.x;
                float y = o.adam_p.y;
                float z = f(x, y);
                DrawSphere({ x, y,z }, 0.05, { 255, 0,255 });
            }

            {
                int n = std::min((int)o.ph.size(), 1000);
                for (int i = 0; i < n; ++i)
                {
                    float x = o.ph[i].x;
                    float y = o.ph[i].y;
                    float z = f(x, y);
                    DrawSphere({ x, y,z }, 0.01, { 0, 128,128 });
                }
                PrimBegin(PrimitiveMode::LineStrip);
                for (int i = 0; i < n; ++i)
                {
                    float x = o.ph[i].x;
                    float y = o.ph[i].y;
                    float z = f(x, y);
                    PrimVertex({ x, y, z }, { 0, 128, 128 });
                }
                PrimEnd();
            }
            {
                int n = std::min((int)o.adam_ph.size(), 100);
                for (int i = 0; i < n; ++i)
                {
                    float x = o.adam_ph[i].x;
                    float y = o.adam_ph[i].y;
                    float z = f(x, y);
                    DrawSphere({ x, y,z }, 0.01, { 128, 0,128 });
                }

                PrimBegin(PrimitiveMode::LineStrip);
                for (int i = 0; i < n; ++i)
                {
                    float x = o.adam_ph[i].x;
                    float y = o.adam_ph[i].y;
                    float z = f(x, y);
                    PrimVertex({ x, y, z }, { 128, 0, 128 });
                }
                PrimEnd();
            }
        }

        // history 
        for (int i = 0; i < points.size(); ++i)
        {
            points[i].ph.push_back(points[i].p);
            points[i].adam_ph.push_back(points[i].adam_p);
        }

        iterations++;

        // optimize SGD
        float sgd_alpha = 0.2f;
        for( int i = 0; i < points.size(); ++i )
        {
            OptimizePoint& o = points[i];
            float x = o.p.x;
            float y = o.p.y;
            o.p = o.p - sgd_alpha * glm::vec2(dfdx(x, y), dfdy(x, y));
        }

        // optimize Adam
        static float alpha = 0.1f;
        static float beta1 = 0.9f;
        static float beta2 = 0.999f;
        static float e = 0.00000001f;
        for (int i = 0; i < points.size(); ++i)
        {
            OptimizePoint& o = points[i];
            float x = o.adam_p.x;
            float y = o.adam_p.y;
            glm::vec2 p = o.adam_p;
            glm::vec2 dp = glm::vec2(dfdx(x, y), dfdy(x, y) );
            o.adam_m = beta1 * o.adam_m + (1.0f - beta1) * dp;
            o.adam_v = beta2 * o.adam_v + (1.0f - beta2) * (dp * dp);
            glm::vec2 adam_m_hat = o.adam_m / ( 1.0f - beta1 );
            glm::vec2 adam_v_hat = o.adam_v / ( 1.0f - beta2 );
            o.adam_p = o.adam_p - alpha * adam_m_hat / ( glm::sqrt( adam_v_hat ) + glm::vec2(e, e) );
        }

        PopGraphicState();
        EndCamera();

        BeginImGui();

        ImGui::SetNextWindowSize({ 500, 800 }, ImGuiCond_Once);
        ImGui::Begin("Panel");
        ImGui::Text("fps = %f", GetFrameRate());
        ImGui::InputInt("nSamples", &nSamples);
        
        if (ImGui::Button("init"))
        {
            initPoints();
        }
        ImGui::Text("iterations = %d", iterations);

        ImGui::End();

        EndImGui();
    }

    pr::CleanUp();
}
