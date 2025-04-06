document.addEventListener(`DOMContentLoaded`, () => {
    let spanDefectNode = document.querySelector(`#defect_level`);
    let spanQualityNode = document.querySelector(`#quality_score`);

    let defectLevel = parseFloat(spanDefectNode.textContent.trim());
    let qualityScore = parseFloat(spanQualityNode.textContent.trim());

    if (defectLevel < 0.3) {
        spanDefectNode.classList.add(`text-success`);
    } else if (defectLevel >= 0.3 && defectLevel < 0.7) {
        spanDefectNode.classList.add(`text-warning`);
    } else if (defectLevel >= 0.7) {
        spanDefectNode.classList.add(`text-danger`);
    }

    if (qualityScore >= 7) {
        spanQualityNode.classList.add(`text-success`);
    } else if (qualityScore >= 4) {
        spanQualityNode.classList.add(`text-warning`);
    } else {
        spanQualityNode.classList.add(`text-danger`);
    }
});
