<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>Helmet Safety Check</title>

    <link rel="stylesheet" href="style.css">
</head>

<body>

<div class="container">

    <!-- HEADER -->
    <header class="header">

        <div class="logo">🪖</div>

        <div>
            <h1>Helmet Safety Check</h1>

            <p>
                AI-powered helmet compliance detection
            </p>
        </div>

    </header>


    <!-- UPLOAD SECTION -->
    <section class="upload-card">

        <h2>Upload Rider Image or Video</h2>

        <p class="description">
            Upload an image or video to check helmet compliance.
        </p>

        <input
            type="file"
            id="imageInput"
            accept=".jpg,.jpeg,.png,.mp4,.avi,.mov"
        >

        <button id="analyzeButton">
            🔍 Analyze
        </button>

        <div
            id="loading"
            class="loading hidden"
        >
            🤖 AI is analyzing...
        </div>

        <div
            id="error"
            class="error hidden"
        ></div>

    </section>


    <!-- VIDEO RESULT -->
    <section
        id="videoResultSection"
        class="hidden"
    >

        <div class="result-card">

            <h2>🎥 Video Analysis</h2>

            <p>
                Video processing completed successfully.
            </p>

            <video
                id="resultVideo"
                controls
                style="width:100%; max-width:900px;"
            ></video>

        </div>

    </section>


    <!-- IMAGE RESULT -->
    <section
        id="resultSection"
        class="hidden"
    >

        <!-- MAIN STATUS -->
        <div
            id="statusCard"
            class="status-card"
        >

            <div
                id="statusIcon"
                class="status-icon"
            >
                🟡
            </div>

            <div>

                <h2 id="statusTitle">
                    RESULT UNCLEAR
                </h2>

                <p id="statusMessage"></p>

            </div>

        </div>


        <!-- METRICS -->
        <div class="metrics">

            <div class="metric-card">

                <span>
                    With Helmet
                </span>

                <strong id="helmetCount">
                    0
                </strong>

            </div>


            <div class="metric-card">

                <span>
                    Without Helmet
                </span>

                <strong id="noHelmetCount">
                    0
                </strong>

            </div>


            <div class="metric-card">

                <span>
                    Licence
                </span>

                <strong id="licenceCount">
                    0
                </strong>

            </div>


            <div class="metric-card">

                <span>
                    Total Detections
                </span>

                <strong id="detectionCount">
                    0
                </strong>

            </div>

        </div>


        <!-- SAFETY STATUS -->
        <div class="result-card">

            <h2>
                🪖 Safety Status
            </h2>

            <div class="detail">

                <span>
                    Status
                </span>

                <strong id="safetyStatus"></strong>

            </div>


            <div class="detail">

                <span>
                    YOLO Confidence
                </span>

                <strong id="confidence"></strong>

            </div>


            <div class="detail">

                <span>
                    Recommended Action
                </span>

                <strong id="recommendation"></strong>

            </div>

        </div>


        <!-- DETECTION DETAILS -->
        <div class="result-card">

            <h2>
                📊 Detection Details
            </h2>

            <div id="detections"></div>

        </div>

    </section>

</div>


<script src="script.js"></script>

</body>

</html>