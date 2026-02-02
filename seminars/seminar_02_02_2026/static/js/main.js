const helloLLM = document.querySelector('h1');
const btn = document.querySelector('button');

btn.addEventListener('click', () => {
    helloLLM.style.color = `#${Math.floor(Math.random()*16777215).toString(16)}`;
});
