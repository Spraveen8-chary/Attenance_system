window.addEventListener("load",()=>{
    const loader = document.querySelector(".loader");

    loader.classList.add("loader-hidden");

    loader.addEventListener("transitionend",() => {
        document.body.removeChild("loader")
    })
})

document.querySelector(".itemLogout").addEventListener("mousedown", function () {
    document.querySelector(".itemLogout").style.backgroundColor = "blue";
})

document.querySelector(".itemLogout").addEventListener("mouseup", function () {
    document.querySelector(".itemLogout").style.backgroundColor = "#5bc0eb";
})