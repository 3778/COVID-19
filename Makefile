include ./help.mk

image_repo=3778
image=$(image_repo)/covid-19:latest

launch:
	streamlit run app.py

data-launch:
	streamlit run data/data_app.py

collect:
	python data/collectors.py

bin/gh-md-toc:
	mkdir -p bin
	wget https://raw.githubusercontent.com/ekalinin/github-markdown-toc/master/gh-md-toc
	chmod a+x gh-md-toc
	mv gh-md-toc bin/

.PHONY: README.md
README.md: bin/gh-md-toc
	./bin/gh-md-toc --insert README.md
	rm -f README.md.orig.* README.md.toc.*

.PHONY: covid-19
covid-19: ## Run covid-19 container
	docker run \
		--rm \
		--publish 8501:8501 \
		--name covid-19 \
		--volume $(shell pwd):/covid-19 \
		$(image)

.PHONY: image
image: ## Build covid-19 image
	docker build . --tag $(image)
