package simple

import (
	"encoding/json"
	"io/ioutil"
	"log"
)

func WritePretty(data interface{}, filename string) {
	bytes, err := json.MarshalIndent(data, "", "    ")
	if err != nil {
		log.Fatalln(err)
	}

	_ = ioutil.WriteFile(filename, bytes, 0755)
}
