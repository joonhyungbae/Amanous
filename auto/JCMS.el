(TeX-add-style-hook
 "JCMS"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "a4paper")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("fontenc" "T1") ("biblatex" "backend=biber" "style=apa") ("graphicx" "pdftex")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "inputenc"
    "fontenc"
    "type1cm"
    "booktabs"
    "amsfonts"
    "nicefrac"
    "microtype"
    "graphicx"
    "csquotes"
    "biblatex"
    "url"
    "hyperref")
   (TeX-add-symbols
    "RR"
    "Nat"
    "CC")
   (LaTeX-add-labels
    "GFI"
    "H"
    "CFTR"
    "sample-table"))
 :latex)

