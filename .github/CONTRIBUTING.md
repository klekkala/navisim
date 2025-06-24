# Contributing Guidelines

Thank you for your interest in contributing to this project. To keep our workflow clean and efficient, please follow the guidelines below.

---

## Branching Strategy

We use a simplified Git Flow model with the following branches:

- `main`: Production-ready, stable code only
- `develop`: Integrates all new features and fixes before release
- `feature/fix/chore/hotfix` branches: Created from `develop` for isolated work

### Branch Types

| Prefix     | Purpose                        | Example                         |
|------------|--------------------------------|---------------------------------|
| `feat/`    | New features                   | `feat/user-authentication`      |
| `fix/`     | Bug fixes                      | `fix/login-crash`               |
| `chore/`   | Non-functional updates         | `chore/clean-logs`              |
| `hotfix/`  | Critical production fixes      | `hotfix/payment-overflow`       |

---

## Creating a Branch

Always branch from `develop`:

```bash
git checkout develop
git pull
git checkout -b feat/my-awesome-feature
```

---

## Pull Requests

- Open all pull requests against the `develop` branch.
- Use appropriate pull request templates (`feature`, `fix`, `chore`, etc.).
- Provide a clear description of the changes, related issue numbers, and screenshots if applicable.
- Assign at least one reviewer.
- Ensure all CI checks pass before requesting a review.

---

## Code Review Requirements

- At least one approved review is required before merging.
- All comments should be addressed before merging.
- Avoid committing directly to `main` or `develop`.
- Commit messages should be clear and descriptive.
- Ensure that all tests pass and the branch is up to date with `develop`.

For more details, refer to the project documentation or contact a maintainer.
