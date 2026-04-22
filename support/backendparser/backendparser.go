// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package backendparser parses the compute interfaces (Backend, Builder, Function, etc.) and enumerate
// their methods.
//
// This is useful to generate code that works with these interfaces.
package backendparser

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"

	"github.com/gomlx/compute/internal/exceptions"
	"github.com/gomlx/compute/internal/must"
)

// Method represents a single method from the backends.Builder or backends.Function interface
// with all its signature information as strings.
type Method struct {
	// Name is the method name
	Name string
	// Comment is the method documentation comment
	Comments []string
	// Parameters of the method.
	Parameters []NameAndType
	// Outputs of the method.
	// Outputs names may contain all empty strings if they are not defined.
	Outputs []NameAndType
	// Interface indicates which interface this method belongs to: "Builder" or "Function"
	Interface string
}

type NameAndType struct {
	Name, Type string
}

// ParseBuilder returns all methods defined in the backends.Builder and backends.Function interfaces,
// including those from embedded interfaces like backends.StandardOps and backends.CollectiveOps.
func ParseBuilder() ([]Method, error) {
	fileSet := token.NewFileSet()
	var methods []Method

	root, err := findModuleRoot()
	if err != nil {
		return nil, err
	}

	// Parse all relevant files
	builderFile, err := parser.ParseFile(fileSet, filepath.Join(root, "builder.go"),
		nil, parser.ParseComments)
	if err != nil {
		return nil, err
	}
	functionFile, err := parser.ParseFile(fileSet, filepath.Join(root, "function.go"),
		nil, parser.ParseComments)
	if err != nil {
		return nil, err
	}
	standardOpsFile, err := parser.ParseFile(fileSet, filepath.Join(root, "standard_ops.go"),
		nil, parser.ParseComments)
	if err != nil {
		return nil, err
	}
	collectiveOpsFile, err := parser.ParseFile(fileSet, filepath.Join(root, "collectiveops.go"),
		nil, parser.ParseComments)
	if err != nil {
		return nil, err
	}
	fusedOpsFile, err := parser.ParseFile(fileSet, filepath.Join(root, "fused_ops.go"),
		nil, parser.ParseComments)
	if err != nil {
		return nil, err
	}

	// File contents cache
	fileCache := make(map[string][]byte)
	getFileContent := func(fileName string) []byte {
		fileContent, ok := fileCache[fileName]
		if !ok {
			// File not in cache, read it
			fileContent = must.M1(os.ReadFile(fileName))
			fileCache[fileName] = fileContent
		}
		return fileContent
	}

	// Extract the text from a node
	getText := func(node ast.Node) string {
		pos := fileSet.Position(node.Pos())
		fileName := pos.Filename
		fileContent := getFileContent(fileName)

		// Extract text from the cached file content
		endOffset := fileSet.Position(node.End()).Offset
		if endOffset > len(fileContent) {
			exceptions.Panicf("end offset out of bounds for file %s", fileName)
		}
		return string(fileContent[pos.Offset:endOffset])
	}

	// Helper to extract methods from interface declarations
	includeInterfaces := []string{"Builder", "Function", "StandardOps", "CollectiveOps", "FusedOps"}
	extractMethods := func(file *ast.File) {
		ast.Inspect(file, func(n ast.Node) bool {
			if typeSpec, ok := n.(*ast.TypeSpec); ok {
				if interfaceType, ok := typeSpec.Type.(*ast.InterfaceType); ok {
					if slices.Index(includeInterfaces, typeSpec.Name.Name) == -1 {
						return true
					}
					for _, method := range interfaceType.Methods.List {
						// Extract method information
						funcType, ok := method.Type.(*ast.FuncType)
						if !ok {
							continue
						}

						m := Method{
							Name:      method.Names[0].Name,
							Interface: typeSpec.Name.Name,
						}

						// Get method comment if any
						if method.Doc != nil {
							m.Comments = make([]string, 0, len(method.Doc.List))
							for _, comment := range method.Doc.List {
								m.Comments = append(m.Comments, comment.Text)
							}
						}

						// Get parameters
						if funcType.Params != nil {
							for _, param := range funcType.Params.List {
								paramType := getText(param.Type)
								for _, name := range param.Names {
									param := NameAndType{Name: name.Name, Type: paramType}
									m.Parameters = append(m.Parameters, param)
								}
							}
						}

						// Get outputs
						if funcType.Results != nil {
							for _, result := range funcType.Results.List {
								resultType := getText(result.Type)
								if len(result.Names) == 0 {
									m.Outputs = append(m.Outputs, NameAndType{Type: resultType})
								} else {
									for _, name := range result.Names {
										param := NameAndType{Name: name.Name, Type: resultType}
										m.Outputs = append(m.Outputs, param)
									}
								}
							}
						}

						methods = append(methods, m)
					}
				}
			}
			return true
		})
	}

	extractMethods(builderFile)
	extractMethods(functionFile)
	extractMethods(standardOpsFile)
	extractMethods(collectiveOpsFile)
	extractMethods(fusedOpsFile)

	return methods, nil
}

// isTargetModule checks if the go.mod contents declare the target module.
func isTargetModule(modBytes []byte, target string) bool {
	for line := range strings.SplitSeq(string(modBytes), "\n") {
		line = strings.TrimSpace(line)
		if after, ok := strings.CutPrefix(line, "module "); ok {
			modName := strings.TrimSpace(after)
			if idx := strings.Index(modName, "//"); idx != -1 {
				modName = strings.TrimSpace(modName[:idx])
			}
			if modName == target {
				return true
			}
		}
	}
	return false
}

// checkGoWork parses the go.work file at the given path and checks if any of
// its use directives point to a directory containing the target module.
func checkGoWork(gowork, moduleName string) (string, error) {
	workBytes, err := os.ReadFile(gowork)
	if err != nil {
		return "", err
	}
	workDir := filepath.Dir(gowork)

	var inUseBlock bool
	for line := range strings.SplitSeq(string(workBytes), "\n") {
		if idx := strings.Index(line, "//"); idx != -1 {
			line = line[:idx]
		}
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		var usePath string
		if inUseBlock {
			if line == ")" {
				inUseBlock = false
				continue
			}
			usePath = strings.Trim(line, "\"'`")
		} else if after, ok := strings.CutPrefix(line, "use "); ok {
			rest := strings.TrimSpace(after)
			if rest == "(" {
				inUseBlock = true
				continue
			} else {
				usePath = strings.Trim(rest, "\"'`")
			}
		} else if line == "use(" || line == "use (" {
			inUseBlock = true
			continue
		}

		if usePath != "" {
			if !filepath.IsAbs(usePath) {
				usePath = filepath.Join(workDir, usePath)
			}
			if modBytes, err := os.ReadFile(filepath.Join(usePath, "go.mod")); err == nil {
				if isTargetModule(modBytes, moduleName) {
					return usePath, nil
				}
			}
		}
	}
	return "", fmt.Errorf("module %s not found in go.work", moduleName)
}

// findModuleRoot returns the absolute path to the module root directory
// for github.com/gomlx/compute.
func findModuleRoot() (string, error) {
	const moduleName = "github.com/gomlx/compute"

	// 1. Check if the current module (as stated in go.mod) is github.com/gomlx/compute
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for d := dir; ; {
		if modBytes, err := os.ReadFile(filepath.Join(d, "go.mod")); err == nil {
			if isTargetModule(modBytes, moduleName) {
				return d, nil
			}
			break // Found a go.mod, but not our target module, stop going up
		}
		parent := filepath.Dir(d)
		if parent == d {
			break
		}
		d = parent
	}

	// 2. Check if any of the paths in go.work refer to github.com/gomlx/compute
	if outWork, err := exec.Command("go", "env", "GOWORK").Output(); err == nil {
		gowork := strings.TrimSpace(string(outWork))
		if gowork != "" {
			if usePath, err := checkGoWork(gowork, moduleName); err == nil {
				return usePath, nil
			}
		}
	}

	// 3. Check if Go has the cached code for github.com/gomlx/compute
	if outCache, err := exec.Command("go", "env", "GOMODCACHE").Output(); err == nil {
		modCache := strings.TrimSpace(string(outCache))
		if modCache == "" {
			if gopath, err := exec.Command("go", "env", "GOPATH").Output(); err == nil {
				modCache = filepath.Join(strings.TrimSpace(string(gopath)), "pkg", "mod")
			}
		}
		if modCache != "" {
			matches, err := filepath.Glob(filepath.Join(modCache, filepath.FromSlash(moduleName)+"@*"))
			if err == nil && len(matches) > 0 {
				slices.Sort(matches)
				return matches[len(matches)-1], nil
			}
		}
	}

	return "", fmt.Errorf("could not find module root for %s (checked current go.mod, go.work, and GOMODCACHE)", moduleName)
}
