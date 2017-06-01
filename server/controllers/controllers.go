package controllers

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"

	"github.com/stubborn-manatee/server/models"
)

type (
	PostController struct{}
)

func NewPostController() *PostController {
	return &PostController{}
}

func (pc PostController) GetPost(w http.ResponseWriter, r *http.Request) {
	fmt.Println("Getting post")
	p := models.Post{
		Id:    "00000",
		Title: "A new beginning",
		Body:  "<p> The day had just begun. <p>",
	}

	pj, err := json.Marshal(p)
	if err != nil {
		log.Print(err)
		w.WriteHeader(http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	fmt.Fprintf(w, "%s", pj)
}

func (pc PostController) CreatePost(w http.ResponseWriter, r *http.Request) {
	p := models.Post{}

	json.NewDecoder(r.Body).Decode(&p)

	p.Id = "00001"

	pj, err := json.Marshal(p)
	if err != nil {
		log.Print(err)
		w.WriteHeader(http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	fmt.Fprintf(w, "%s", pj)
}
