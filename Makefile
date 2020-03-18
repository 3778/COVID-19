.PHONY: README.md

launch:
	streamlit run app.py

bin/gh-md-toc:
	mkdir -p bin
	wget https://raw.githubusercontent.com/ekalinin/github-markdown-toc/master/gh-md-toc
	chmod a+x gh-md-toc
	mv gh-md-toc bin/

README.md: bin/gh-md-toc
	./bin/gh-md-toc --insert README.md
	rm -f README.md.orig.* README.md.toc.*
