# Changelog

## [0.1.1](https://github.com/google/adk-go/compare/plugin/agentanalytics/v0.1.0...plugin/agentanalytics/v0.1.1) (2026-11-17)


### Bug Fixes

* **agentanalytics:** cap response summary and function name summary length ([#2246](https://github.com/google/adk-go/issues/2246)) ([f55660a](https://github.com/google/adk-go/commit/f55660a66749dd7ffc80a25f4ac6c64bdb890817))
* **agentanalytics:** preserve retired late parent cancel cause and custom metadata truncation flag ([#2020](https://github.com/google/adk-go/issues/2020)) ([ca197aa](https://github.com/google/adk-go/commit/ca197aaaa582e9c6d85da4559729a099a7c98907))
* **agentanalytics:** redact and bound code part labels in summaries ([#2332](https://github.com/google/adk-go/issues/2332)) ([cf55159](https://github.com/google/adk-go/commit/cf55159d2d7894e682fb006267910a8ded36ccb6)), closes [#2259](https://github.com/google/adk-go/issues/2259)
* **agentanalytics:** redact credential-named types and guard omitzero in JSONFormatter ([#2249](https://github.com/google/adk-go/issues/2249)) ([5b3d897](https://github.com/google/adk-go/commit/5b3d897b19898a31df19ab786ee85670e69e9a61))
* **agentanalytics:** sanitize unsafe GCSPathPrefix values and cap GCS object names at 1,024 bytes ([#2018](https://github.com/google/adk-go/issues/2018)) ([11f4e29](https://github.com/google/adk-go/commit/11f4e29f588ee0f08d2e372f76737412075460bf))


### Documentation

* **agentanalytics:** add package godoc documenting Go design and parity behavior ([#2176](https://github.com/google/adk-go/issues/2176)) ([1edb057](https://github.com/google/adk-go/commit/1edb0572264d71b3ba3ab4e9bd24a979e267a96e))
