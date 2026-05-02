// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package main

import (
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path"
	"regexp"
	"slices"
	"strings"
	"text/template"

	"github.com/gomlx/compute/internal/must"
	"k8s.io/klog/v2"
)

type DTypeInfo struct {
	DType, GoType string
}

type MapInfo struct {
	MapName, Generic string
	DTypes           []DTypeInfo
}

type MapPairInfo struct {
	MapName, Generic string
	DTypes1, DTypes2 []DTypeInfo
}

type Data struct {
	Package         string
	ImportGobackend bool
	PriorityPrefix  string
	Maps            []MapInfo
	PairMaps        []MapPairInfo
}

var (
	fileName = "gen_dtypemaps_registration.go"

	reDTypeMap     = regexp.MustCompile(`//\s*gobackend:dtypemap\s+(\w+)\s+([\w,]+)`)
	reDTypePairMap = regexp.MustCompile(`//\s*gobackend:dtypemap_pair\s+(\w+)\s+([\w,]+)\s+([\w,]+)`)

	dtypesBFloat16 = []DTypeInfo{{"BFloat16", "bfloat16.BFloat16"}}
	dtypesFloat16  = []DTypeInfo{{"Float16", "float16.Float16"}}
)

func makeDTypes(ints, uints, floats, halfPrecision, boolean, packed bool) []DTypeInfo {
	dtypes := make([]DTypeInfo, 0, 32)
	if ints {
		dtypes = append(dtypes,
			DTypeInfo{"Int8", "int8"},
			DTypeInfo{"Int16", "int16"},
			DTypeInfo{"Int32", "int32"},
			DTypeInfo{"Int64", "int64"},
		)
	}
	if uints {
		dtypes = append(dtypes,
			DTypeInfo{"Uint8", "uint8"},
			DTypeInfo{"Uint16", "uint16"},
			DTypeInfo{"Uint32", "uint32"},
			DTypeInfo{"Uint64", "uint64"},
		)
	}
	if floats {
		dtypes = append(dtypes,
			DTypeInfo{"Float32", "float32"},
			DTypeInfo{"Float64", "float64"},
		)
	}
	if halfPrecision {
		dtypes = append(dtypes,
			DTypeInfo{"BFloat16", "bfloat16.BFloat16"},
			DTypeInfo{"Float16", "float16.Float16"},
		)
	}
	if boolean {
		dtypes = append(dtypes,
			DTypeInfo{"Bool", "bool"},
		)
	}
	if packed {
		dtypes = append(dtypes,
			DTypeInfo{"Int2", "uint8"},
			DTypeInfo{"Int4", "uint8"},
			DTypeInfo{"Uint2", "uint8"},
			DTypeInfo{"Uint4", "uint8"},
		)
	}
	return dtypes
}

func parseTypeGroups(groupsStr string) []DTypeInfo {
	groupsStr = strings.TrimSpace(groupsStr)
	if groupsStr == "bf16" {
		return dtypesBFloat16
	}
	if groupsStr == "f16" {
		return dtypesFloat16
	}
	var ints, uints, floats, half, boolean, packed bool
	parts := strings.Split(groupsStr, ",")
	for _, p := range parts {
		p = strings.TrimSpace(p)
		switch p {
		case "ints":
			ints = true
		case "uints":
			uints = true
		case "floats":
			floats = true
		case "half":
			half = true
		case "bool":
			boolean = true
		case "packed":
			packed = true
		}
	}
	return makeDTypes(ints, uints, floats, half, boolean, packed)
}

func parseFiles(dir string) (Data, error) {
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, dir, func(fi os.FileInfo) bool {
		return !strings.HasPrefix(fi.Name(), "gen_") && strings.HasSuffix(fi.Name(), ".go")
	}, parser.ParseComments)
	if err != nil {
		return Data{}, err
	}

	var d Data
	for pkgName, pkg := range pkgs {
		if strings.HasSuffix(pkgName, "_test") {
			continue
		}
		d.Package = pkgName
		if pkgName != "gobackend" {
			d.ImportGobackend = true
			d.PriorityPrefix = "gobackend."
		}
		for _, file := range pkg.Files {
			for _, decl := range file.Decls {
				genDecl, ok := decl.(*ast.GenDecl)
				if !ok || genDecl.Tok != token.VAR {
					continue
				}

				var comments []string
				if genDecl.Doc != nil {
					for _, c := range genDecl.Doc.List {
						comments = append(comments, c.Text)
					}
				}

				for _, spec := range genDecl.Specs {
					vSpec, ok := spec.(*ast.ValueSpec)
					if !ok {
						continue
					}

					var vComments []string
					vComments = append(vComments, comments...)
					if vSpec.Doc != nil {
						for _, c := range vSpec.Doc.List {
							vComments = append(vComments, c.Text)
						}
					}

					for _, name := range vSpec.Names {
						mapName := name.Name
						for _, comment := range vComments {
							if match := reDTypeMap.FindStringSubmatch(comment); match != nil {
								generic := match[1]
								typeGroups := match[2]
								d.Maps = append(d.Maps, MapInfo{
									MapName: mapName,
									Generic: generic,
									DTypes:  parseTypeGroups(typeGroups),
								})
							} else if match := reDTypePairMap.FindStringSubmatch(comment); match != nil {
								generic := match[1]
								typeGroups1 := match[2]
								typeGroups2 := match[3]
								d.PairMaps = append(d.PairMaps, MapPairInfo{
									MapName: mapName,
									Generic: generic,
									DTypes1: parseTypeGroups(typeGroups1),
									DTypes2: parseTypeGroups(typeGroups2),
								})
							}
						}
					}
				}
			}
		}
	}

	return d, nil
}

func main() {
	klog.InitFlags(nil)
	flag.Parse()

	data, err := parseFiles(".")
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to parse files: %v\n", err)
		os.Exit(1)
	}

	// Deterministic order for generated code.
	slices.SortFunc(data.Maps, func(a, b MapInfo) int {
		return strings.Compare(strings.ToUpper(a.MapName), strings.ToUpper(b.MapName))
	})
	slices.SortFunc(data.PairMaps, func(a, b MapPairInfo) int {
		return strings.Compare(strings.ToUpper(a.MapName), strings.ToUpper(b.MapName))
	})

	registerTemplate := template.Must(
		template.
			New(fileName).
			Parse(
				`/***** File generated by ./internal/cmd/gobackend_dtypemap. Don't edit it directly. *****/

package {{.Package}}

import (
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
{{- if .ImportGobackend}}
	"github.com/gomlx/compute/internal/gobackend"
{{- end}}
)


func init() {
{{$prefix := .PriorityPrefix}}
{{- range .Maps}}

	// DTypeMap: {{.MapName}}
{{- $mapName := .MapName }}
{{- $generic := .Generic }}
{{- range .DTypes }}
	{{$mapName}}.Register(dtypes.{{.DType}}, {{$prefix}}PriorityGeneric, {{$generic}}[{{.GoType}}])
{{- end }}
{{- end }}

{{- range .PairMaps}}

	// DTypePairMap: {{.MapName}}
{{- $mapName := .MapName }}
{{- $generic := .Generic }}
{{- $dtypes2 := .DTypes2 }}
{{- range .DTypes1 }}
{{- $dtype1 := .DType }}
{{- $goType1 := .GoType }}
{{- range $dtypes2 }}
	{{$mapName}}.Register(dtypes.{{$dtype1}}, dtypes.{{.DType}}, {{$prefix}}PriorityGeneric, {{$generic}}[{{$goType1}}, {{.GoType}}])
{{- end }}
{{- end }}
{{- end }}

}
`))
	fullPath := path.Join(must.M1(os.Getwd()), fileName)
	f := must.M1(os.Create(fullPath))
	must.M(registerTemplate.Execute(f, data))
	must.M(f.Close())

	cmd := exec.Command("gofmt", "-w", fullPath)
	klog.V(1).Infof("\t%s\n", cmd)
	must.M(cmd.Run())
	fmt.Printf("✅ gobackend_dtypemap:  \tsuccessfully generated %s\n", fullPath)
}
