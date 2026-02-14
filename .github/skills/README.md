# ONNX GitHub Copilot Skills

This directory contains GitHub Copilot skills - reusable guides and checklists that help developers perform common tasks in the ONNX repository.

## Available Skills

### [Update ONNX Operator](update-onnx-operator.md)
A comprehensive checklist for updating an existing ONNX operator to a new version. Use this when making breaking changes to an operator's signature, behavior, or supported types.

**When to use:**
- Adding/removing/renaming operator attributes
- Modifying operator inputs or outputs
- Changing operator behavior
- Adding support for new data types

## How to Use Skills

Skills are reference documents that provide:
- Step-by-step checklists
- File locations and code examples
- Best practices and conventions
- Links to related documentation

You can reference these skills when:
- Working on operator updates
- Reviewing pull requests
- Onboarding new contributors
- Ensuring consistency across changes

## Contributing

To add a new skill:
1. Create a new markdown file named `SKILL.md` in this directory (lowercase with hyphens)
2. Add YAML frontmatter with required fields: `name`, `description`, and optionally `license`
3. Follow the structure of existing skills
4. Include clear checklists and examples
5. Add links to relevant documentation
6. Update this README with the new skill

See [GitHub's skill documentation](https://docs.github.com/en/copilot/concepts/agents/about-agent-skills) for details on the skill file format.

## Related Documentation

- [Adding New Operators](../../docs/AddNewOp.md)
- [Updating Operators](../../docs/UpdatingOperator.md)
- [ONNX Versioning](../../docs/Versioning.md)
- [Version Converter](../../docs/VersionConverter.md)
