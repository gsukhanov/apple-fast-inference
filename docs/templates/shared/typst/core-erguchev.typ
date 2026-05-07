#import "core-base.typ": create-core

#let core = create-core(yaml("../../term-paper-erguchev.yaml"))

#let project-name() = (core.project_name)()
#let project-name-english() = (core.project_name_english)()
#let project-summary() = (core.project_summary)()
#let project-type() = (core.project_type)()
#let paragraph(body) = (core.paragraph)(body)
#let render-document(id, body) = (core.render_document)(id, body)
