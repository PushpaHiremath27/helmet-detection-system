const imageInput = document.getElementById("imageInput");
const analyzeButton = document.getElementById("analyzeButton");
const loading = document.getElementById("loading");
const errorBox = document.getElementById("error");
const resultSection = document.getElementById("resultSection");

const statusCard = document.getElementById("statusCard");
const statusIcon = document.getElementById("statusIcon");
const statusTitle = document.getElementById("statusTitle");
const statusMessage = document.getElementById("statusMessage");

const helmetCount = document.getElementById("helmetCount");
const noHelmetCount = document.getElementById("noHelmetCount");
const licenceCount = document.getElementById("licenceCount");
const detectionCount = document.getElementById("detectionCount");

const safetyStatus = document.getElementById("safetyStatus");
const confidence = document.getElementById("confidence");
const recommendation = document.getElementById("recommendation");

const detections = document.getElementById("detections");


analyzeButton.addEventListener("click", async function () {

    errorBox.classList.add("hidden");
    resultSection.classList.add("hidden");

    const file = imageInput.files[0];

    if (!file) {
        showError("Please select an image first.");
        return;
    }

    const formData = new FormData();

    formData.append("image", file);
    formData.append("confidence", "0.40");
    formData.append("iou", "0.45");

    loading.classList.remove("hidden");

    analyzeButton.disabled = true;
    analyzeButton.textContent = "Analyzing...";


    try {

        const response = await fetch(
            "/api/detect",
            {
                method: "POST",
                body: formData
            }
        );


        if (!response.ok) {

            throw new Error(
                "Server returned HTTP " +
                response.status
            );

        }


        const data = await response.json();


        if (!data.success) {

            throw new Error(
                data.error ||
                "Detection failed."
            );

        }


        displayResult(data);


    } catch (error) {

        console.error(
            "Detection error:",
            error
        );

        showError(
            "Detection failed: " +
            error.message
        );


    } finally {

        loading.classList.add("hidden");

        analyzeButton.disabled = false;

        analyzeButton.textContent =
            "🔍 Analyze Image";
    }

});


function displayResult(data) {

    resultSection.classList.remove("hidden");


    const counts =
        data.class_counts || {};

    const safety =
        data.safety || {};

    const gemini =
        data.gemini || {};


    // -----------------------------------------
    // YOLO COUNTS
    // -----------------------------------------

    helmetCount.textContent =
        counts["With Helmet"] || 0;

    noHelmetCount.textContent =
        counts["Without Helmet"] || 0;

    licenceCount.textContent =
        counts["licence"] || 0;

    detectionCount.textContent =
        data.detection_count || 0;


    // -----------------------------------------
    // SAFETY STATUS
    // -----------------------------------------

    statusIcon.textContent =
        safety.icon || "🟡";

    statusTitle.textContent =
        safety.status ||
        "RESULT UNCLEAR";

    statusMessage.textContent =
        safety.message || "";


    safetyStatus.textContent =
        safety.safety_status ||
        "Requires Review";


    recommendation.textContent =
        safety.recommendation ||
        "Please review manually.";


    // -----------------------------------------
    // YOLO CONFIDENCE
    // -----------------------------------------

    let highestConfidence = 0;

    const detectionList =
        data.detections || [];


    detectionList.forEach(
        function (item) {

            if (
                item.class === "With Helmet" ||
                item.class === "Without Helmet"
            ) {

                highestConfidence =
                    Math.max(
                        highestConfidence,
                        Number(item.confidence) || 0
                    );
            }
        }
    );


    confidence.textContent =
        (
            highestConfidence * 100
        ).toFixed(1) + "%";


    // -----------------------------------------
    // DETECTION DETAILS
    // -----------------------------------------

    renderDetections(
        detectionList
    );


    // -----------------------------------------
    // GEMINI VERIFICATION
    // -----------------------------------------

    renderGeminiVerification(
        data,
        gemini
    );


    // -----------------------------------------
    // STATUS CARD
    // -----------------------------------------

    if (
        safety.status ===
        "NO HELMET DETECTED"
    ) {

        statusCard.style.borderLeft =
            "6px solid #dc2626";

    } else if (
        safety.status ===
        "HELMET DETECTED"
    ) {

        statusCard.style.borderLeft =
            "6px solid #16a34a";

    } else {

        statusCard.style.borderLeft =
            "6px solid #ca8a04";
    }

}


function renderGeminiVerification(
    data,
    gemini
) {

    // Remove previous Gemini card
    const oldCard =
        document.getElementById(
            "geminiCard"
        );

    if (oldCard) {
        oldCard.remove();
    }


    const card =
        document.createElement("div");

    card.id = "geminiCard";
    card.className =
        "result-card";


    const title =
        document.createElement("h2");

    title.textContent =
        "🤖 AI Verification";


    card.appendChild(title);


    // -----------------------------------------
    // AI VERIFICATION
    // -----------------------------------------

    const verification =
        document.createElement("div");

    verification.className =
        "detail";


    const verificationLabel =
        document.createElement("span");

    verificationLabel.textContent =
        "Verification";


    const verificationValue =
        document.createElement("strong");


    if (
        data.ai_verification ===
        "AI VERIFIED"
    ) {

        verificationValue.textContent =
            "✅ AI VERIFIED";

    } else if (
        data.ai_verification ===
        "HUMAN REVIEW RECOMMENDED"
    ) {

        verificationValue.textContent =
            "🟡 HUMAN REVIEW RECOMMENDED";

    } else {

        verificationValue.textContent =
            "🔵 YOLO HIGH CONFIDENCE";
    }


    verification.appendChild(
        verificationLabel
    );

    verification.appendChild(
        verificationValue
    );

    card.appendChild(
        verification
    );


    // -----------------------------------------
    // GEMINI STATUS
    // -----------------------------------------

    const geminiStatus =
        document.createElement("div");

    geminiStatus.className =
        "detail";


    const geminiStatusLabel =
        document.createElement("span");

    geminiStatusLabel.textContent =
        "Gemini Verification";


    const geminiStatusValue =
        document.createElement("strong");


    geminiStatusValue.textContent =
        gemini.verification ||
        "SKIPPED";


    geminiStatus.appendChild(
        geminiStatusLabel
    );

    geminiStatus.appendChild(
        geminiStatusValue
    );

    card.appendChild(
        geminiStatus
    );


    // -----------------------------------------
    // GEMINI HELMET STATUS
    // -----------------------------------------

    const helmetStatus =
        document.createElement("div");

    helmetStatus.className =
        "detail";


    const helmetStatusLabel =
        document.createElement("span");

    helmetStatusLabel.textContent =
        "Gemini Helmet Status";


    const helmetStatusValue =
        document.createElement("strong");

    helmetStatusValue.textContent =
        gemini.helmet_status ||
        "UNCLEAR";


    helmetStatus.appendChild(
        helmetStatusLabel
    );

    helmetStatus.appendChild(
        helmetStatusValue
    );

    card.appendChild(
        helmetStatus
    );


    // -----------------------------------------
    // GEMINI CONFIDENCE
    // -----------------------------------------

    const geminiConfidence =
        document.createElement("div");

    geminiConfidence.className =
        "detail";


    const geminiConfidenceLabel =
        document.createElement("span");

    geminiConfidenceLabel.textContent =
        "Gemini Confidence";


    const geminiConfidenceValue =
        document.createElement("strong");


    const confidenceValue =
        Number(
            gemini.confidence
        ) || 0;


    geminiConfidenceValue.textContent =
        (
            confidenceValue * 100
        ).toFixed(1) + "%";


    geminiConfidence.appendChild(
        geminiConfidenceLabel
    );

    geminiConfidence.appendChild(
        geminiConfidenceValue
    );

    card.appendChild(
        geminiConfidence
    );


    // -----------------------------------------
    // GEMINI REASON
    // -----------------------------------------

    const reason =
        document.createElement("div");

    reason.className =
        "detail";


    const reasonLabel =
        document.createElement("span");

    reasonLabel.textContent =
        "Gemini Reason";


    const reasonValue =
        document.createElement("strong");

    reasonValue.textContent =
        gemini.reason ||
        "No additional verification reason.";


    reason.appendChild(
        reasonLabel
    );

    reason.appendChild(
        reasonValue
    );

    card.appendChild(
        reason
    );


    // -----------------------------------------
    // INSERT CARD
    // -----------------------------------------

    resultSection.appendChild(
        card
    );
}


function renderDetections(items) {

    detections.innerHTML = "";


    if (items.length === 0) {

        detections.innerHTML =
            "<p>No objects detected.</p>";

        return;
    }


    items.forEach(
        function (item, index) {

            const div =
                document.createElement(
                    "div"
                );

            div.className =
                "detection-item";


            const className =
                document.createElement(
                    "strong"
                );

            className.textContent =
                "Detection " +
                (index + 1);


            const classText =
                document.createElement(
                    "div"
                );

            classText.textContent =
                "Class: " +
                item.class;


            const confidenceText =
                document.createElement(
                    "div"
                );

            confidenceText.textContent =
                "Confidence: " +
                (
                    (
                        Number(
                            item.confidence
                        ) || 0
                    ) * 100
                ).toFixed(1) +
                "%";


            div.appendChild(
                className
            );

            div.appendChild(
                classText
            );

            div.appendChild(
                confidenceText
            );


            detections.appendChild(
                div
            );

        }
    );
}


function showError(message) {

    errorBox.textContent =
        message;

    errorBox.classList.remove(
        "hidden"
    );
}