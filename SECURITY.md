# Security

EMShrimulator Studio opens files that other people made: Gerber and Excellon
exports, ODB++ archives, and exported projects. A file that makes the
application run code, or read or write outside the folders it should, is a
security bug.

## Reporting a vulnerability

Report it privately through GitHub: open the repository's **Security** tab and
choose **Report a vulnerability**. Please do not open a public issue for it.

Include the version (Help → About), what you opened or ran, and what happened.
A file that reproduces the problem is the most useful thing you can attach.

## Supported versions

Fixes go into the latest release.

## How untrusted input is handled

- **Project files:** expressions go through a whitelisted evaluator, never
  `eval`; names are checked and resolved paths must stay inside the projects
  folder.
- **Project archives and Gerber archives:** members with `..` are skipped, or
  flattened to their file names.
- **ODB++ archives:** extracted with Python's `data` filter.
- **Gerber aperture macros:** parsed as plain arithmetic (numbers, `$n`
  variables, `+ - x /` and parentheses); anything else is refused.

The solver itself is openEMS, run as a separate process.
