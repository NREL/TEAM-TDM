# TEAM-TDM

This is a package for easy application of multiple machine learning models to a single problem, and comparison via various metrics.
For a pretty detailed description of the models that are available in the tool, and a quite detailed description of the metrics that it pumps out, see [the paper](https://github.com/NREL/TEAM-TDM/blob/master/paper/team_tdm.pdf).

## Installation

1) Clone the repo.
2) Put all of your code into the `src/` folder
3) Import modules as `import ml_battery.some_modules` or `from ml_battery import *`
4) TODO: Create a `setup.py` for farill installation

## Usage

Probably the easiest thing to do is to just copy one of the existing jupyter notebooks, and repurpose it with your own data.

* You have to test/train split your own data, and input a codebook identifying categorical features.  
  See the `fuel_use.ipynb` for a good example of reading in a csv and pumping it into the pipeline
* You can edit items in the model, but you can just run it as-is for first-pass results.
* As such, the model, fitting and scoring lines can all just be run, without editing for new datasets.  

### Logging

Because all of this stuff has the ability to run multiple processes, it imports a handy `log to a socket` functionality from `ml_battery.log`.
In order to log to a socket, there needs to be a logger reading from that socket.
Fortunately, the `src/ml_battery/logging_server.py` script is exactly that.
Run the `logging_server.py` script from anywhere, and a file will be created in the working directory called `test.log` that logs all of the output from the `ml_battery` functions.

## Docs

There is a sphinx documentation framework here.  To build it, go into the docs/ folder and `$make html` (or `./make html` on windows)
This will create a bunch of handy documentation of the individual functions and classes available in the ml_battery library.

<!--

Github Flavored Markdown (GFMD) is based on [Markdown Syntax Guide](http://daringfireball.net/projects/markdown/syntax) with some overwriting as described at [Github Flavored Markdown](http://github.github.com/github-flavored-markdown/)

## Text Writing
It is easy to write in GFMD. Just write simply like text and use the below simple "tagging" to mark the text and you are good to go!  

To specify a paragraph, leave 2 spaces at the end of the line

## Headings

```
# Sample H1
## Sample H2
### Sample H3
```

will produce
# Sample H1
## Sample H2
### Sample H3

---

## Horizontal Rules

Horizontal rule is created using `---` on a line by itself.

---

## Coding - Block

<pre>
```ruby
# The Greeter class
class Greeter
  def initialize(name)
    @name = name.capitalize
  end

  def salute
    puts "Hello #{@name}!"
  end
end

# Create a new object
g = Greeter.new("world")

# Output "Hello World!"
g.salute
```
</pre>
 
will produce  

```ruby
# The Greeter class
class Greeter
  def initialize(name)
    @name = name.capitalize
  end

  def salute
    puts "Hello #{@name}!"
  end
end

# Create a new object
g = Greeter.new("world")

# Output "Hello World!"
g.salute
```

Note: You can specify the different syntax highlighting based on the coding language eg. ruby, sh (for shell), php, etc  
Note: You must leave a blank line before the `\`\`\``

## Coding - In-line
You can produce inline-code by using only one \` to enclose the code:

```
This is some code: `echo something`
```

will produce  

This is some code: `echo something`

---

## Text Formatting
**Bold Text** is done using `**Bold Text**`  
*Italic Text* is done using `*Italic Text*`

---

## Hyperlinks
- GFMD will automatically detect URL and convert them to links like this http://www.futureworkz.com
- To specify a link on a text, do this:

```
This is [an example](http://example.com/ "Title") inline link.
[This link](http://example.net/) has no title attribute.
```

---

## Escape sequence
You can escape using \\ eg. \\\`

---

## Creating list

Adding a `-` will change it into a list:

```
- Item 1
- Item 2
- Item 3
```

will produce

- Item 1
- Item 2
- Item 3

---

## Quoting

You can create a quote using `>`:

```
> This is a quote
```

will produce

> This is a quote

## Table and Definition list

These two can only be created via HTML:

````html
<table>
  <tr>
    <th>ID</th><th>Name</th><th>Rank</th>
  </tr>
  <tr>
    <td>1</td><td>Tom Preston-Werner</td><td>Awesome</td>
  </tr>
  <tr>
    <td>2</td><td>Albert Einstein</td><td>Nearly as awesome</td>
  </tr>
</table>

<dl>
  <dt>Lower cost</dt>
  <dd>The new version of this product costs significantly less than the previous one!</dd>
  <dt>Easier to use</dt>
  <dd>We've changed the product so that it's much easier to use!</dd>
</dl>
```

will produce

<table>
  <tr>
    <th>ID</th><th>Name</th><th>Rank</th>
  </tr>
  <tr>
    <td>1</td><td>Tom Preston-Werner</td><td>Awesome</td>
  </tr>
  <tr>
    <td>2</td><td>Albert Einstein</td><td>Nearly as awesome</td>
  </tr>
</table>

<dl>
  <dt>Lower cost</dt>
  <dd>The new version of this product costs significantly less than the previous one!</dd>
  <dt>Easier to use</dt>
  <dd>We've changed the product so that it's much easier to use!</dd>
</dl>

## Adding Image

```
![Branching Concepts](http://git-scm.com/figures/18333fig0319-tn.png "Branching Map")
```

-->