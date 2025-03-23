document.addEventListener(`DOMContentLoaded`, () => {
    let spanNode = document.querySelector(`#span`)
    let resultText = spanNode.textContent.trim();
    if (resultText == `Большой, довольно качественный`) {
        spanNode.classList.add(`text-success`);
    } else if (resultText == `Большой, плохого качества`) {
        spanNode.classList.add(`text-warning`);
    } else if (resultText == `Маленький, среднего или низкого качества`) {
        spanNode.classList.add(`text-danger`);
    }
});