/*
Copyright [2017] [Elre Oldewage]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
*/

//Package main registers an HTTP handler at "/" that handles incoming requests for stubborn-manatee.
package main

import (
	"flag"
	"fmt"
	_ "fmt"
	"log"
	"net/http"

	"github.com/gorilla/mux"
	ctrls "github.com/stubborn-manatee/server/controllers"
)

//Every handler should be wrapped by this error handler
func errorHandler(f func(http.ResponseWriter, *http.Request) error) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		err := f(w, r)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			log.Printf("Handling %q: %v", r.RequestURI, err)
		}
	}
}

func redirectToHttps(w http.ResponseWriter, r *http.Request) {
	http.Redirect(w, r, "https://"+r.URL.Host+r.RequestURI, http.StatusMovedPermanently)
}

func main() {
	contentPath := flag.String("c", "../../content/", "Path to the content directory")
	serverPort := flag.Int("p", 8080, "Port on which server will be hosted")
	flag.Parse()
	r := mux.NewRouter()
	postController := ctrls.NewPostController()
	r.HandleFunc("/post", postController.GetPost).Methods("GET")
	r.HandleFunc("/post", postController.CreatePost).Methods("POST")
	r.PathPrefix("/").Handler(http.FileServer(http.Dir(*contentPath)))

	http.HandleFunc( "/", redirectToHttps)

	go http.ListenAndServe(":80", nil)
	fmt.Println("Hosting stubborn-manatee on port", fmt.Sprintf("%v", *serverPort))
	err := http.ListenAndServeTLS(fmt.Sprintf(":%v", *serverPort), "/etc/letsencrypt/live/stubborn-manatee.co.za/cert.pem", "/etc/letsencrypt/live/stubborn-manatee.co.za/privkey.pem", r)
	if err != nil {
		panic(err);
	}
}
