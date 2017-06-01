package models

type (
	Post struct {
		Id    string `json:"id"`
		Title string `json:"title"`
		Body  string `json:"body"`
	}
)

/*

	Tags        []string
	PublishDate string
	EditDate    string
	Author      string
	ImgPreview  string
	Blurb       string

*/
