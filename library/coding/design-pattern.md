# Design Pattern
In professional software development, *design patterns* are introduced to make programs easier to maintain and scale. For applications that involve a front-end graphical interface, one common design pattern is Model‑View‑Controller (MVC), where each component has clear responsibilities. Please check reference materials below or do some research on your own.


## Refactoring
It is understandable that adding a design pattern might feel like extra work in course projects. A common practice in software development is to first build a prototype to get basic functionality working, and then refactor the code later to apply design patterns once the structure becomes clearer.

In software development, *refactoring* means improving the internal structure of existing code without changing what the program does. It is a common housekeeping practice for codebases, especially in large‑scale projects involving many developers. As a codebase grows and more features are added, developers often realize that their initial designs for interfaces or components are not ideal for future extension or maintenance. That is where refactoring comes in: developers review the existing code, propose better class structures or interfaces, reorganize functions, or simply clean up messy sections to make the code easier to understand, test, and maintain.

### Examples
1. Rewrite a long and messy “spaghetti” function into a set of smaller utility functions. Each smaller function has one clear job and can be tested independently. This makes the code easier to debug and maintain.

2. If you notice similar logic, including repeated constant values, appearing in multiple classes or functions, you can introduce a common utility function and reuse it in those places. This reduces duplication and makes future updates easier because you only need to change the logic in one location.

## References
1. MVC pattern: [link](https://www.geeksforgeeks.org/system-design/mvc-design-pattern/).
2. Apply MVC to javascript: [link](https://www.sitepoint.com/mvc-design-pattern-javascript/)