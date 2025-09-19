window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"], ["\\begin{equation}", "\\end{equation}"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  // },
  // startup: {
  //   ready: () => {
  //     MathJax.startup.defaultReady();
  //     MathJax.startup.promise.then(() => {
  //       MathJax.typesetPromise();
  //     });
  //   }
  }
};